#!/usr/bin/env bash

#  Install via `conda` directly.
#  This will fail to install all
#  dependencies. If one fails,
#  all dependencies will fail to install.
#
conda install --yes --file requirements

#
#  To go around issue above, one can
#  iterate over all lines in the
#  requirements.txt file.
#
while read requirement; do conda install --yes $requirement; done < requirements

#
#  Error often occurs when a package is not available.
#  Below, in case a package is not available through conda,
#  it will take it through pip
#
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements