.. MethylNet documentation master file, created by
   sphinx-quickstart on Sun Dec  9 18:45:51 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MethylNet's documentation!
=====================================

.. image:: methylnet_pipeline.png
   :width: 800px
   :height: 200px
   :alt: MethylNet
   :align: center

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. click:: preprocess:preprocess
   :prog: python preprocess.py
   :show-nested:

.. click:: embedding:embed
   :prog: python embedding.py
   :show-nested:

.. click:: visualizations:visualize
   :prog: python visualizations.py
   :show-nested:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
