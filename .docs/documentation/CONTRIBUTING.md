# Contributing to [Project]

Thank you for your interest in contributing to this project! We’re excited to have you on board. This guide is designed
to make the contribution process clear and efficient.

---

## Table of Contents

1. [How to Contribute?](#how-to-contribute)
    - [Reporting Issues](#reporting-issues)
    - [Suggesting Features](#suggesting-features)
    - [Coding Standards](#coding-standards)
    - [File Structure](#file-structure)
    - [Commit Message Guidelines](#commit-message-guidelines)
    - [Branches](#branches)
    - [File Naming Convention](#file-naming-convention)
    - [Versioning, Release, Tag](#versioning-release-tag)

---

## How to Contribute?

### Reporting Issues

- If you find a bug, please open a new issue on the [GitHub Issues](https://github.com/[username]/[project-name]/issues)
  page.
- Include the following details:
    - A clear and descriptive title.
    - Steps to reproduce the issue.
    - Expected vs. actual behavior.
    - Screenshots or error logs if applicable.

### Suggesting Features

- Have a great idea for a new feature? Open a new **Issue** and describe your suggestion.
- Explain how this feature will improve the project.

### Coding Standards

##### Import Order

Follow the import order below to maintain consistency in the codebase:

1. **Standard library imports**
2. **Third-party imports**
3. **Local application/library specific imports**

**Example:**

```python
# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from my_module import my_function
```

> **INFO**
>
> *For more detail, please check*:
> - [PEP 8 – Style Guide for Python Code | peps.python.org](https://peps.python.org/pep-0008/)

##### Docstring

Use `NumPy` format for docstrings in both functions and classes:

- Each function or class should have a docstring explaining its purpose, parameters, return values, and examples if
  applicable.
- For classes, include a class-level docstring that describes the overall purpose of the class, any parameters for
  `__init__`, and details on attributes and methods.
- If you include references (e.g., research papers, algorithms, or external resources) in your docstring, create a
  separate `References` section clearly listing these sources at the end of your docstring.

Example for a function:

```python
# Standard library imports
from typing import Annotated


def example_function(
    param1: Annotated[int, "Description of param1"],
    param2: Annotated[str, "Description of param2"]
) -> Annotated[bool, "Description of the return value"]:
    """
    Brief description of what the function does.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of the return value.

    Examples
    --------
    >>> example_function(5, 'hello')
    True
    >>> example_function(0, '')
    False

    References
    ----------
    * Doe, John. "A Study on Example Functions." Journal of Examples, 2021.
    """
    return bool(param1) and bool(param2)
```

Example for a class:

```python
class MyClass:
    """
    MyClass demonstrates the use of docstrings with a separate References section.

    This class provides an example of structuring docstrings, including attributes,
    methods, usage examples, and a references section when external sources are cited.

    Parameters
    ----------
    param1 : str
        Description of `param1`, explaining its purpose and specific constraints.
    param2 : int, optional
        Description of `param2`. Defaults to 0.

    Attributes
    ----------
    attribute1 : str
        Description of `attribute1`, explaining its purpose and possible values.
    attribute2 : int
        Description of `attribute2`, outlining constraints or expected values.

    Methods
    -------
    example_method(param1, param2=None)
        Example method description.

    Examples
    --------
    Create an instance and use methods:

    >>> my_instance = MyClass("example", 5)
    >>> my_instance.example_method("sample")

    References
    ----------
    * Smith, Jane. "Guide to Effective Python Documentation." Python Publishing, 2020.
    * Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-Excitation Networks."
      IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.
    """

    def __init__(self, param1, param2=0):
        """
        Initializes the class.

        Parameters
        ----------
        param1 : str
            Explanation of `param1`.
        param2 : int, optional
            Explanation of `param2`. Defaults to 0.
        """
        self.attribute1 = param1
        self.attribute2 = param2

    @staticmethod
    def example_method(param1, param2=None):
        """
        Example method.

        Parameters
        ----------
        param1 : str
            Description of param1.
        param2 : int, optional
            Description of param2. Defaults to None.

        Returns
        -------
        bool
            Outcome of the method's action.
        """
        return bool(param1) and bool(param2)
```

This structure ensures clear, consistent, and comprehensive documentation of classes and functions, including proper
citation of external sources.

##### Type Annotation

Add type annotations to all functions using `Annotated` with descriptions:

Example:

```python
# Standard library imports
from typing import Annotated


def calculate_area(
    radius: Annotated[float, "Radius of the circle"]
) -> Annotated[float, "Area of the circle"]:
    """
    Calculate the area of a circle given its radius.

    Parameters
    ----------
    radius : float
        Radius of the circle.

    Returns
    -------
    float
        Area of the circle.

    Examples
    --------
    >>> calculate_area(5)
    78.53999999999999
    >>> calculate_area(0)
    0.0
    """
    if not isinstance(radius, (int, float)):
        raise TypeError("Expected int or float for parameter 'radius'")
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return 3.1416 * radius ** 2

```

##### Type Check

Add type check within functions to ensure the correctness of input parameters:

Example:

```python
# Standard library imports
from typing import Annotated

def add_numbers(
    a: Annotated[int, "First integer"],
    b: Annotated[int, "Second integer"]
) -> Annotated[int, "Sum of a and b"]:
    """
    Add two integers and return the result.

    Parameters
    ----------
    a : int
        First integer.
    b : int
        Second integer.

    Returns
    -------
    int
        The sum of `a` and `b`.

    Examples
    --------
    >>> add_numbers(2, 3)
    5
    >>> add_numbers(-1, 5)
    4
    """
    if not isinstance(a, int):
        raise TypeError("Expected int for parameter 'a'")
    if not isinstance(b, int):
        raise TypeError("Expected int for parameter 'b'")
    return a + b
```

##### Doctest

Include doctest examples in docstrings using the `>>>` format:

Example:

```python
# Standard library imports
from typing import Annotated

def multiply(
    a: Annotated[int, "First integer"],
    b: Annotated[int, "Second integer"]
) -> Annotated[int, "Product of a and b"]:
    """
    Multiply two integers and return the result.

    Parameters
    ----------
    a : int
        First integer.
    b : int
        Second integer.

    Returns
    -------
    int
        The product of `a` and `b`.

    Examples
    --------
    >>> multiply(2, 3)
    6
    >>> multiply(-1, 5)
    -5

    This is a doctest example.
    """
    if not isinstance(a, int):
        raise TypeError("Expected int for parameter 'a'")
    if not isinstance(b, int):
        raise TypeError("Expected int for parameter 'b'")

    return a * b
```

> **INFO**
>
> *For more detail, please check*:
> - [doctest — Test interactive Python examples — Python 3.13.1 documentation](https://docs.python.org/3/library/doctest.html)

##### Main Execution

Add each file to `Name-Main` code script:

Example:

```python
# Standard library imports
from typing import Annotated

def example_function(
    x: Annotated[int, "An integer parameter"]
) -> Annotated[int, "The square of x"]:
    """
    Calculate the square of an integer.

    Parameters
    ----------
    x : int
        The integer to be squared.

    Returns
    -------
    int
        The square of `x`.

    Examples
    --------
    >>> example_function(5)
    25
    >>> example_function(-3)
    9
    """
    return x * x

if __name__ == "__main__":
    value = 5
    result = example_function(value)
    print(f"The square of {value} is {result}.")
```

##### General

- **Always Print Outputs to the Terminal**
    - Ensure that any significant results or status messages are displayed to the user via `print` statements.
    - Consider adding an optional parameter (e.g., `verbose=True`) that controls whether to print the outputs. This way,
      users can disable or enable printed outputs as needed.

- **Reduce Code Complexity if It Does Not Disrupt the Flow**
    - Whenever possible, simplify or refactor functions, methods, and classes.
    - Clear, straightforward logic is easier to maintain and less error-prone.

- **Keep Your Code Modular at All Times**
    - Break down larger tasks into smaller, reusable functions or modules.
    - Modular design improves readability, promotes code reuse, and simplifies testing and maintenance.

- **Use Base Classes if Classes Become Too Complex**
    - If a class starts to grow unwieldy or complicated, consider extracting shared logic into a base (parent) class.
    - Child classes can inherit from this base class, reducing duplication and making the code more organized.

### File Structure

Follow the [Default Project Template](https://github.com/bunyaminergen/DefaultProjectTemplate)'s File Structure

- Adhere to the predetermined file hierarchy and naming conventions defined in the Default Project Template.
- Review the existing layout in the repository to ensure your contributions align with the project’s organization.

### Commit Message Guidelines

- **Keep It Short and Concise:**
    - The subject line (summary) should typically not exceed **50 characters**.
    - If more details are needed, include them in the body of the message on a separate line.

- **Use Present Tense and Imperative Mood:**
    - **Start your commit message with only of the following verbs** and then explain what you did:
        - `Add`
        - `Fix`
        - `Remove` or `Delete`
        - `Update`
        - `Test`
        - `Refactor`
    - Messages should use the present tense and imperative mood.
    - **Examples:**
        - `Add user authentication`
        - `Fix bug in payment processing`
        - `Remove unused dependencies`

- **Separate Subject and Details:**
    - The first line (subject) should be short and descriptive.
    - Leave a blank line between the subject and the detailed description (if needed).
        - **Example:**

        ```text
        Fix login issue

        Updated the authentication service to handle null values in the session token.
        ```

- **Mistakes to Avoid:**
    - **Vague Messages:**
        - *Bad Example:* `Fix stuff`, `Update files`, `Work done`.
    - **Combining Multiple Changes in One Commit:**
        - Avoid bundling unrelated changes into a single commit.
    - **Copy-Paste Descriptions:**
        - Ensure that the commit message is directly relevant to the change.

- **Benefits of Good Commit Messages:**
    - A well-written commit history makes the project easier to understand.
    - It simplifies debugging and troubleshooting.
    - It improves collaboration within the team by providing clear and meaningful information.

### Branches

To maintain consistency across all branches, follow these guidelines:

- **Start with one of the following action keywords in lowercase:**
    - `add`
    - `fix`
    - `remove` or `delete`
    - `update`
    - `test`
    - `refactor`
- Use hyphens (`-`) to separate words in the branch name.
- Avoid special characters, spaces, or uppercase letters.
- Keep branch names concise but descriptive.

**Example Branch Names:**

- `add-new-release`
- `fix-critical-bug`
- `remove-unused-dependencies`
- `update-api-endpoints`
- `test-api-performance`
- `refactor-code-structure`

Please push all development work to the `develop` branch. Once the work on your branch is finished, merge it into
`develop` and then delete the branch to keep the repository clean.

**Important:** Please only create branches that begin with the prefixes listed below. If you would like to propose a new
prefix, kindly open an issue on GitHub.

##### Bug Branches

Use the `bugfix/` prefix for bug fixes discovered during development or testing.  
Examples:

- `bugfix/fix-typo-in-readme`
- `bugfix/null-pointer-exception`

##### Feature Branches

Use the `feature/` prefix for new features or enhancements.  
Examples:

- `feature/add-login`
- `feature/update-dashboard`
- `feature/fix-bug-123`

##### Hotfix Branches

Use the `hotfix/` prefix for critical fixes that need immediate attention in production.  
Example:

- `hotfix/fix-security-issue`

##### Docfix Branches

Use the `docfix/` prefix for changes regarding documentation.  
Example:

- `docfix/add-readme-to-artitecture-section`

##### Test Branches

Use the `test/` prefix for branches that focus on writing or updating tests, or conducting specific test-related work.  
Examples:

- `test/add-integration-tests`
- `test/refactor-unit-tests`
- `test/performance-testing`

##### Experiment Branches

Use the `experiment/` prefix for experimental or proof-of-concept work.  
Example:

- `experiment/improve-cache`

---

### File Naming Convention

This section explains how packages and modules should be named in this project.

---

##### Package & Module Naming

The following rules apply to **both** packages and modules. To simplify these guidelines, the term **"file"** will be
used as a generic reference to either a package or a module name.

- **Single or Concise Compound Words:**
    - Whenever possible, each file name should consist of **a single word** or **a short concatenation of words** (
      written together).
    - Use **lowercase letters** only, and **do not** use underscores (`_`).  
      *(Although PEP 8 allows snake_case for modules, we do not prefer it in this project.)*
    - If more than one word is necessary, write them **together** (e.g., `datautils`, `webparser`).

- **Consistent Single-Word Usage:**
    - Especially for package names, aim to keep them to **one word** whenever possible. If multiple words are necessary,
      the package name should remain **short and clear**.

- **Parent (Package) and Sub-Component (Module) Logic:**
    - Use broader, more general names for packages (the “parent”).
    - For modules (the “child”), use more specific names within the package to reflect their functionality.

- **Examples:**
    - **Packages**
        - `utils`
        - `model`
    - **Modules**
        - `gridsearch.py`
        - `convolution.py`
    - **Parent–Child (Directory Structure) Example**
        - `src/model/backbone.py`
        - `src/utils/log/manager.py`

- **Bad Examples:**
    - **Packages**
        - `data_reader`  *(underscores are discouraged)*
    - **Modules**
        - `grid_search.py`  *(underscores are discouraged)*
    - **Parent–Child (Directory Structure) Example**
        - `src/train/training.py`  *(names are too similar or redundant)*

---

##### Test Files

1. **Using the `test_` Prefix:**
    - For test files, prepend `test_` to the **module** name being tested.
    - **Example:** If the module name is `dataprocess.py`, then the test file should be named `test_dataprocess.py`.

> **INFO**  
> *For more details, please check:*
> - [PEP 8 – Style Guide for Python Code | peps.python.org](https://peps.python.org/pep-0008/#:~:text=Modules%20should%20have%20short%2C%20all,use%20of%20underscores%20is%20discouraged)
> - [PEP 423 – Naming conventions and recipes related to packaging | peps.python.org](https://peps.python.org/pep-0423/#follow-pep-8-for-syntax-of-package-and-module-names)

---

### Versioning, Release, Tag

- When numbering versions, releases, or tags, **only use prime numbers** (e.g., 3, 5, 7, 11, 13...).
- **Do not** use 2 (even though it is prime) or any non-prime numbers (4, 6, 8, 9, 10...).

#### Examples

**Good Examples** (using only prime numbers, excluding 2):

- **Tag**: `v3`
- **Release**: `v3.5.7`
- **Version**: `5.7.11`

**Bad Examples**:

- **Tag**: `v2` *(2 is prime but disallowed in this project)*
- **Release**: `v1.2.3` *(1 is not prime, 2 is disallowed, 3 is prime but the rest are invalid)*
- **Version**: `4.6.8` *(4, 6, 8 are not prime)*

---

