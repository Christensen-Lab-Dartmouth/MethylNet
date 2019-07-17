from setuptools import setup
with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()
setup(name='methylnet',
      version='0.1.2',
      description='A modular deep learning approach for Methylation Predictions.',
      url='https://github.com/Christensen-Lab-Dartmouth/MethylNet',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=['bin/install_methylnet_dependencies','bin/download_help_data'],
      entry_points={
            'console_scripts':['methylnet-embed=methylnet.embedding:embed',
                               'methylnet-predict=methylnet.predictions:prediction',
                               'methylnet-interpret=methylnet.model_interpretability:interpret',
                               'methylnet-visualize=methylnet.visualizations:visualize',
                               'methylnet-torque=methylnet.torque_job_runner:torque',
                               'methylnet-test=methylnet.test_pipeline:test']
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['methylnet'],
      install_requires=['pymethylprocess',
                        'ruamel.yaml',
                        'modin',
                        'statsmodels',
                        'mlxtend'])
