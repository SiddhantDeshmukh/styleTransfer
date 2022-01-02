using Flux, Metalhead, Autograd, Images

# TODO
# - functions to load and preprocess images to specified size
# - functions to init the models

function normalise_img!(img::Array{Float64}, model_mean::Float)
  # Subtract mean of images in training dataset from the img
  for i in 1:size(img, 3)  # image is 3-Tensor (height, width, colour)
    img[:,:,i] = img[:,:,i] - model_mean[i]
  end
  return img
end

function preprocess(path::String; new_size=256)
  img = load(path)
  # Resize so larger dimension scaled to new_size, preserving height/width
  # ratio
  height = size(img, 1)
  width = size(img, 2)
  if max(height, width) == new_size
    img2 = img
  elseif height >= width
    img2 = imresize(img, new_size, Int(round(new_size * width/height)))
  else
    img2 =  imresize(img, Int(round(new_size * width/height)), new_size)
  end
  # Convert img2 to 3D Tensor [height, width, colour]
  img2 = channelview(img2)  # [C, H, W]
  img2 = permutedims(img2, [2,3,1])  # [H, W, C]
  img2 = Array{Float64}(img2)
  # Normalise by subtracting VGG19 model mean
  model_mean = Array{Float64}(averageImage ./ 255)
  normalise_img!(img, model_mean)
  # Add extra dimension at end
  img3 = reshape(img2, (size(img2)...), 1)  # [H,W,C,1]
  return img3
end

function postprocess(img::Array{Float64})
  # Convert image tensor into image object
  img = reshape(img, (size(img)...)[1:end-1])  # [H,W,C,1] -> [H,W,C]
  # Unnormalise img by adding model mean
  model_mean = Array{Float64}(average_image ./ 255)
  normalise_img!(img, -model_mean)
  # Clamp tensor to make it valid Image object
  clamp!(img, 0, 1)
  img = Array{FixedPointNumbers.Normed{Uint8, 8}}(img)
  return colorview(RGB, permutedims(img, [3,1,2]))  # [C,H,W]
end

function load_dataset(path::String, batch::Int, total::Int)
  z = readdir(path)
  indices = randperm(length(z))[1:total]
  paths = [joinpath(path, i) for i in z[indices]]
  images = []
  for (counts, file_path) in enumerate(paths)
    img = preprocess(file_path; new_size=224)
    # ndims(img) == 3 ? push!(images, img) : total -= 1
    push!(images, img)
    counts % 100 == 0 && info("$counts images loaded")
  end
  return [cat(4, images[i]...) for i in partition(1:total, batch)]
end

# Loss functions and respective utilities
function content_loss(content_weight, current_features, original_features)
  H1, W1, C1, Q = size(features)
  content_losses = content_weight * (sum((current_features - original_features).^2) / (4*H1*W1*C1))
  return content_losses
end

function gram_matrix(features; normalise=true)
  H,W,C,N = size(features)
  feat_reshaped = reshape(features, (H*W, C))
  gram_mat = feat_reshaped' * feat_reshaped  # [C,C]
  return normalise ? gram_mat ./ (2*H*W*C) : gram_mat
end

function style_loss(features, style_layers, style_targets, style_weights)
  style_losses = Float64(0.0)
  for i in 1:length(style_layers)
    gram_mat = gram_matrix(features[style_layers[i]])
    style_losses = style_losses + style_weights[i] * sum((gram_mat - style_targets[i]).^2)
  end
  return style_losses
end

# Custom layers
# ConvPad
ConvPad(chs::Pair{<:Int, <:Int}, kernel::Tuple{Int,Int}; stride::Tuple{Int,Int}=(1,1)) =
  Conv(kernel, chs, stride=stride, pad=(kernel[1]÷2, kernel[2]÷2))

# Residual Block
struct ResidualBlock
  conv_layers
  norm_layers
end

ResidualBlock(chs::Int) = 
  ResidualBlock(((Conv(3,3), chs=>chs, pad=(1,1)), Conv((3,3), chs=>chs, pad=(1,1))), (BatchNorm(chs), BatchNorm(chs)))

# Upsampling Block
Upsample(x) = repeat(x, inner=(2,2,1,1))

UpsamplingBlock(chs::Pair{<:Int, <:Int}, kernel::Tuple{Int, Int}, stride::Tuple{Int, Int}, upsample::Int, pad::Tuple{Int, Int} = (0,0)) = 
  Chain(Conv(kernel, chs, stride=stride, pad=(kernel[1]÷2, kernel[2]÷2)), x->Upsample(x))

# Transformer model
Transformer() = Chain(
              # Conv layers            
              ConvPad(3=>32, (3,3)),
              BatchNorm(32, relu),
              ConvPad(32=>64, (3,3), stride=(2,2)),
              BatchNorm(64, relu),
              ConvPad(64=>128, (3,3), stride=(2,2)),
              BatchNorm(128, relu),
              # Residual Blocks
              [ResidualBlock(128) for i in 1:5]...,
              # Upsampling
              UpsamplingBlock(128=>64, (3,3), (1,1), 2)
              BatchNorm(64),
              UpsamplingBlock(64=>32, (3,3), (3,3), (1,1), 2)
              BatchNorm(32),
              ConvPad(32=>3, (9,9), stride=(1,1))
)

# VGG19 - Remove classification Dense layers at the end
mutable struct vgg19
    slice1
    slice2
    slice3
    slice4
end

function vgg19()
    vgg = VGG19().layers[1]
    slice1 = Chain(vgg[1:5]...)
    slice2 = Chain(vgg[6:10]...)
    slice3 = Chain(vgg[11:15]...)
    slice4 = Chain(vgg[16:20]...)
    vgg19(slice1, slice2, slice3, slice4)
end

function (layer::vgg19)(x)
    res1 = layer.slice1(x)
    res2 = layer.slice2(res1)
    res3 = layer.slice3(res2)
    res4 = layer.slice4(res3)
    (res1, res2, res3, res4)
end

# Average image that VGG19 was trained on
average_image = [123.68, 116.779, 103.939]  # RGB

function train(train_data_path, batch_size, η, style_image_path, epochs, model_save_path, content_weight, style_weight, model=Transformer; images=10000)
  train_dataset = load_dataset(train_data_path, batch_size, images)
  optimizer = ADAM(params(model), η)
  style = preprocess(style_image_path; new_size=224)
  style = repeat(reshape(style, size(style)..., 1), outer=(1,1,1,batch_size))
  vgg = vgg19()
  features_style = vgg(style)
  gram_style = [gram_matrix(y) for y in features_style]
end