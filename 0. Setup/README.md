# 0. Setup

In this first chapter all that was required was to create a python virtual environment : 

```bash
python3 -m venv .venv
```

As well as running the following command :

```bash
pip install "transformers[sentencepiece]"
```

However not all of that worked. The first issue was that torch did not work with python3.13, which is the currently newest version. Because of that, python3.10 was required :

```bash
brew install python@3.10
```

Once that was done, we removed the old virtual environment and created a new one.

Afterwards, it was necessary to install all the packages :

```bash
pip install torch transformers numpy
```

And it still didn't work. That is because torch does not support numpy2.0 . So, it was necessary to install an older numpy version : 

```bash
pip install "numpy<2"
```

Finally, everything worked.