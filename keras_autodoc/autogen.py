import shutil
import pathlib
from inspect import getdoc, isclass
from typing import Dict, Union, List, get_type_hints

from .docstring import process_docstring
from .examples import copy_examples
from .get_signatures import get_signature

from . import utils


class DocumentationGenerator:
    """Generates the documentation.

    # Arguments

        pages: A dictionary. The keys are the files' paths, the values
            are lists of strings, functions /classes / methods names
            with dotted access to the object. For example,
            `pages = {'my_file.md': ['keras.layers.Dense']}` is valid.
            Values can also be dictionaries, introducing an extra level to map
            custom tags to lists of elements to be documented.
        project_url: The url pointing to the module directory of your project on
            GitHub. This will be used to make a `[Sources]` link.
        template_dir: Where to put the markdown files which will be copied and
            filled in the destination directory. You should put files like
            `index.md` inside. If you want a markdown file to be filled with
            the docstring of a function, use the `{{autogenerated}}` tag inside,
            and then add the markdown file to the `pages` dictionary.
        example_dir: Where you store examples in your project. Usually standalone
            files with a markdown docstring at the top. Will be inserted in the docs.
        extra_aliases: When displaying type hints, it's possible that the full
            dotted path is displayed instead of alias. The aliases present in
            `pages` are used, but it may happen if you're using a third-party library.
            For example `tensorflow.python.ops.variables.Variable` is displayed instead
            of `tensorflow.Variable`. Here you have two solutions, either you provide
            the import keras-autodoc should follow:
            `extra_aliases=["tensorflow.Variable"]`, either you provide a mapping to use
            `extra_aliases={"tensorflow.python.ops.variables.Variable": "tf.Variable"}`.
            The second option should be used if you want more control and that you
            don't want to respect the alias corresponding to the import (you can't do
            `import tf.Variable`). When giving a list, keras-autodoc will try to import
            the object from the string to understand what object you want to replace.
        max_signature_line_length: When displaying class and function signatures,
            keras-autodoc formats them using Black. This parameter controls the
            maximum line length of these signatures, and is passed directly through
            to Black.
        titles_size: `"#"` signs to put before a title in the generated markdown.
    """
    def __init__(self,
                 pages: Dict[str, list] = {},
                 project_url: Union[str, Dict[str, str]] = None,
                 template_dir=None,
                 examples_dir=None,
                 extra_aliases: Union[List[str], Dict[str, str]] = None,
                 max_signature_line_length: int = 110,
                 titles_size="###"):
        self.pages = pages
        self.project_url = project_url
        self.template_dir = template_dir
        self.examples_dir = examples_dir
        self.class_aliases = {}
        self._fill_aliases(extra_aliases)
        self.max_signature_line_length = max_signature_line_length
        self.titles_size = titles_size

    def generate(self, dest_dir):
        """Generate the docs.

        # Arguments

            dest_dir: Where to put the resulting markdown files.

        # Raises
            TypeError: if elements are not specified as list or dict of lists.
        """
        dest_dir = pathlib.Path(dest_dir)
        print("Cleaning up existing sources directory.")
        if dest_dir.exists():
            shutil.rmtree(dest_dir)

        print("Populating sources directory with templates.")
        if self.template_dir:
            shutil.copytree(self.template_dir, dest_dir)

        for file_path, elements in self.pages.items():
            if isinstance(elements, list):
                self._render_list_and_insert(elements, dest_dir / file_path)
            elif isinstance(elements, dict):
                for tag, grouped_elements in elements.items():
                    if isinstance(grouped_elements, list):
                        self._render_list_and_insert(
                            grouped_elements, dest_dir / file_path, tag)
                    else:
                        raise TypeError(
                            "Expected list of elements to be documented, is of type {}: {}"
                            .format(type(grouped_elements), grouped_elements))
            else:
                raise TypeError(
                    "Expected list of elements to be documented, is of type {}: {}"
                    .format(type(elements), elements))

        if self.examples_dir is not None:
            copy_examples(self.examples_dir, dest_dir / "examples")

    def process_docstring(self, docstring, types: dict = None):
        """Can be overridden."""
        processsed = process_docstring(docstring, types, self.class_aliases)
        return processsed

    def process_signature(self, signature):
        """Can be overridden."""
        return signature

    def _render(self, element):
        if isinstance(element, str):
            object_ = utils.import_object(element)
            if utils.ismethod(object_) or isinstance(object_, property):
                # we remove the modules when displaying the methods
                signature_override = '.'.join(element.split('.')[-2:])
            else:
                signature_override = element
        else:
            signature_override = None
            object_ = element

        return self._render_from_object(object_, signature_override)

    def _render_list_and_insert(self, element_list, file_path, tag="autogenerated"):
        markdown_text = ''
        for element in element_list:
            markdown_text += self._render(element)
        utils.insert_in_file(markdown_text, file_path, tag)

    def _render_from_object(self, object_, signature_override: str):
        subblocks = []
        if self.project_url is not None:
            if isinstance(object_, property):
                subblocks.append(utils.make_source_link(object_.fget, self.project_url))
            else:
                subblocks.append(utils.make_source_link(object_, self.project_url))
        signature = get_signature(
            object_, signature_override, self.max_signature_line_length
        )
        signature = self.process_signature(signature)

        if not isinstance(object_, property):
            subblocks.append(f"{self.titles_size} {object_.__name__}\n")
            subblocks.append(utils.code_snippet(signature))
        else:
            object_ = object_.fget
            subblocks.append(f"{self.titles_size} {object_.__name__}\n")

        docstring = getdoc(object_)
        if docstring:
            if isclass(object_):
                type_hints = get_type_hints(object_.__init__)
            else:
                type_hints = get_type_hints(object_)
            docstring = self.process_docstring(docstring, type_hints)
            subblocks.append(docstring)
        return "\n\n".join(subblocks) + '\n\n----\n\n'

    def _fill_aliases(self, extra_aliases):
        for elements in self.pages.values():
            if isinstance(elements, dict):
                # collect all lists of elements to be documented
                list_elements = []
                for element in elements.values():
                    list_elements += element
            else:
                list_elements = elements
            for element_as_str in list_elements:
                element = utils.import_object(element_as_str)
                if not isclass(element):
                    continue
                true_dotted_path = utils.get_dotted_path(element)
                self.class_aliases[true_dotted_path] = element_as_str

        if isinstance(extra_aliases, dict):
            self.class_aliases.update(extra_aliases)
        elif isinstance(extra_aliases, list):
            for alias in extra_aliases:
                full_dotted_path = utils.get_dotted_path(utils.import_object(alias))
                self.class_aliases[full_dotted_path] = alias
