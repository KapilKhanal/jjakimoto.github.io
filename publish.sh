git add -A
git commit -m"New publication"
git push origin develop
pelican content -s publishconf.py
ghp-import output -b master
git push origin master
