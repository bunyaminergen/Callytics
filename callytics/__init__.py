import warnings
import importlib.resources as pkg_resources
import importlib.metadata as importlib_metadata

warnings.resetwarnings()
warnings.simplefilter("always")

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"

def get_resource_path(package, resource):
    return str(pkg_resources.files(package).joinpath(resource))