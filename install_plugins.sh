# Install Ipython plugins
# Remove if there is already the repository
rm -rf plugins/ipynb
git rm -rf plugins/ipynb
git submodule add --force git://github.com/danielfrg/pelican-ipynb.git plugins/ipynb

# Install Pelican plugins
# Remove if there is already the repository
rm -rf pelican-plugins
git rm -rf pelican-plugins
git submodule add --force git@github.com:getpelican/pelican-plugins.git
