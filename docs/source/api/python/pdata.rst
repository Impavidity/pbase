Pdata -- Dataset processing library of pbase
============================================

Json Object Naming Convention
-----------------------------

#. **tokens**: Denotes tokenized sentence
#. **tags**: Denotes token level annotation
#. **sentence**: Denotes a sentence string. Tokens are separated by space.
#. **label**: Denotes sentence level label, just one label.


Dataset Preprocessing
---------------------

#. **Ontonotes**:

.. code:: bash

    python -m pbase.pdata.cli ontonotes_ner --file_path /path/to/train/data/language/annotations --dump_path /path/to/data.train
