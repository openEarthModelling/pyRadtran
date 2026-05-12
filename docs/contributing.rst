Contributing
============

Please see the `CONTRIBUTING.md <https://github.com/openEarthModelling/pyRadtran/blob/main/CONTRIBUTING.md>`_ file in the repository for detailed contribution guidelines.

Quick Reference
---------------

Development Setup
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/openEarthModelling/pyRadtran.git
    cd pyRadtran
    pip install -e ".[dev]"

Running Tests
^^^^^^^^^^^^^

.. code-block:: bash

    pytest tests/ -v

Running Tests with Coverage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pytest tests/ --cov=pyradtran --cov-report=html

Linting
^^^^^^^

.. code-block:: bash

    ruff check src/ tests/
    ruff format --check src/ tests/

Building Documentation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd docs
    make html
