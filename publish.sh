git add -A
git commit -m"New publication"
pelican content -s publishconf.py
ghp-import output -b master
git push origin master
