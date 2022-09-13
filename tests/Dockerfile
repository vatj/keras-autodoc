FROM python:3.7 as base_image

RUN pip install tensorflow
RUN pip install markdown mkdocs pytest sphinx

FROM base_image as integration_tests
COPY ./ ./keras-autodoc
RUN pip install -e "./keras-autodoc[tests]"
WORKDIR keras-autodoc
RUN pytest tests/test_integration.py

FROM base_image as unit_tests_py
COPY ./ ./keras-autodoc
RUN pip install -e ./keras-autodoc
WORKDIR keras-autodoc
RUN pytest --ignore=tests/test_integration.py tests/

FROM base_image as flake8_tests
RUN pip install flake8
COPY ./ ./keras-autodoc
WORKDIR keras-autodoc
RUN flake8 .

FROM base_image as doc_tests
COPY ./ ./keras-autodoc
RUN pip install -e keras-autodoc
WORKDIR keras-autodoc/docs
RUN python autogen.py
RUN mkdocs build

FROM base_image as test_release
COPY ./ ./keras-autodoc
RUN pip install ./keras-autodoc
