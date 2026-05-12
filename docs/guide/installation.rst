Installation
============

Prerequisites
-------------

pyRadtran requires **libRadtran** to be installed separately. libRadtran is not bundled with this package.

Installing libRadtran
^^^^^^^^^^^^^^^^^^^^^

Download and install libRadtran from the `official website <https://www.libradtran.org>`_ or GitHub::

    wget https://github.com/rayference/libradtran/archive/refs/tags/v2.0.6.tar.gz
    tar -xzf v2.0.6.tar.gz
    cd libradtran-2.0.6
    ./configure --prefix=/usr/local
    make
    sudo make install

After installation, ensure the ``uvspec`` binary is on your ``PATH``, or set one of these environment variables::

    export LIBRADTRANDIR=/usr/local/share/libRadtran
    # or
    export LIBRADTRAN_DATA_FILES=/usr/local/share/libRadtran/data

Installing pyRadtran
--------------------

From PyPI
^^^^^^^^^

.. code-block:: bash

    pip install pyradtran

For plotting support (matplotlib and scipy)::

    pip install pyradtran[plot]

For development (includes test and documentation dependencies)::

    pip install pyradtran[dev]

From Source
^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/openEarthModelling/pyRadtran.git
    cd pyRadtran
    pip install -e ".[dev]"

Verifying Installation
----------------------

.. code-block:: python

    import pyradtran
    print(pyradtran.__version__)

    from pyradtran import Scene, Runner
    print("pyRadtran imported successfully")
