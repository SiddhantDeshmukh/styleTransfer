# Add, commit and push to GitHub with supplied message

read -p "Confirm: Add everything and push to GitHub? (y/n)" yn
case $yn in
  [Yy]* )
    # Check for commit message
    if [[ "$1" != "" ]]; then
      MESSAGE="$1"
    else
      MESSAGE=$(date +"%Y-%m-%d %T")
      echo "No commit message supplied, using date time $MESSAGE"
    fi
    git add .
    git commit -m "$MESSAGE"
    git push
    ;;
  [Nn]* ) exit;;
  * ) echo "Please enter (y/n)"
esac