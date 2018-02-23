#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals


# Default setting
AUTHOR = 'Tomoaki Fujii'
SITENAME = 'Data Rounder'
SITESUBTITLE = 'Machine Learning, Finance, and Technologies'
SITEURL = 'http://jjakimoto.github.io'

# pusblish an article as a draft in default setting
DEFAULT_METADATA = {
    'status': 'draft',
}

PATH = 'content'
STATIC_PATHS=['images']
PAGE_PATHS = ['pages']
ARTICLE_PATHS = ['articles']


THEME = "/home/tomoaki/work/DS_blog/mytheme"
# THEME = "/Users/tomoaki/work/blog/pelican-themes/bootlex"

# PELICAN_COMMENT_SYSTEM = True
# PELICAN_COMMENT_SYSTEM_IDENTICON_DATA = ('author',)
DISQUS_SITENAME = "datarounder"

TIMEZONE = 'Asia/Tokyo'
DATE_FORMATS={
        'en': '%a, %m/%d/%Y',
}
LOCALE=('en_US')

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_DOMAIN = SITEURL
FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/%s.atom.xml'
# TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = 'feeds/%s.atom.xml'
# AUTHOR_FEED_RSS = None

# Blogroll
# LINKS = (('GitHub', 'https://github.com/jjakimoto'),)

# GITHUB_URL = 'https://github.com/jjakimoto'

# Social widget
SOCIAL = (('GitHub', 'https://github.com/jjakimoto'),
          ('LinkedIn', 'https://www.linkedin.com/in/tomoaki-fujii-5bba56133'),
          ('Facebook', 'https://www.facebook.com/tomoaki.fujii.73'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True

MARKUP = ('md', 'ipynb')

PLUGIN_PATHS = ['./plugins', './pelican-plugins']
PLUGINS = ['ipynb.markup', 'pelican_comment_system', 'assets', 'sitemap', 'gravatar']


DISPLAY_PAGES_ON_MENU = True

ARTICLE_URL = 'articles/{date:%Y}/{date:%b}/{date:%d}/{slug}/'
ARTICLE_SAVE_AS = 'articles/{date:%Y}/{date:%b}/{date:%d}/{slug}/index.html'
PAGE_URL = 'pages/{slug}/'
PAGE_SAVE_AS = 'pages/{slug}/index.html'

# COMMENT_URL = "#my_own_comment_id_{slug}"
