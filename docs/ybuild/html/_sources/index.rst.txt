Welcome to MethylNet's documentation!
=====================================

https://github.com/Christensen-Lab-Dartmouth/MethylNet

See README.md in Github repository for install directions and for example scripts for running the pipeline (not all datasets may be available on GEO at this time).

There is both an API and CLI available for use. Examples for CLI usage can be found in ./example_scripts.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. image:: yimages/10.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. image:: yimages/11.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. image:: yimages/12.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. image:: yimages/13.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. automodule:: methylnet.datasets
   :members:

.. automodule:: methylnet.hyperparameter_scans
   :members:

.. automodule:: methylnet.interpretation_classes
   :members:

.. automodule:: methylnet.models
   :members:

.. automodule:: methylnet.plotter
   :members:

.. automodule:: methylnet.schedulers
   :members:

.. click:: methylnet.embedding:embed
  :prog: methylnet-embed
  :show-nested:

.. click:: methylnet.predictions:predict
 :prog: methylnet-predict
 :show-nested:

.. click:: methylnet.visualizations:visualize
:prog: methylnet-visualize
:show-nested:

.. click:: methylnet.torque_job_runner:torque
:prog: methylnet-torque
:show-nested:

  ipts':['methylnet-embed=methylnet.embedding:embed',
                           'methylnet-predict=methylnet.predictions:predict',
                           'methylnet-interpret=methylnet.model_interpretability:interpret',
                           'methylnet-visualize=methylnet.visualizations:visualize',
                           'methylnet-torque=methylnet.torque_job_runner:torque']
  },
Example Usage
=============

.. image:: yimages/1.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. image:: yimages/2.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. image:: yimages/3.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. image:: yimages/4.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. image:: yimages/5.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. image:: yimages/6.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. image:: yimages/7.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. image:: yimages/8.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

.. image:: yimages/9.jpeg
   :width: 800px
   :height: 600px
   :scale: 60%
   :alt: Download
   :align: center

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
