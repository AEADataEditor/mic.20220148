###############################
### re-use of packages
# The following lines are for folks re-using this package/ replicating the project

using Pkg

# activate the report's quasi-package
Pkg.activate(".")
# instantiate: this will install any new dependencies in Manifest.toml and Project.toml
Pkg.instantiate()
