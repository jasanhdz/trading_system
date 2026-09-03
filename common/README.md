# Common

This directory is reserved for versioned code shared by two or more sandbox
projects. It must not import from `sandbox/` and must not contain experiment
configuration, artifacts, or runtime state.

No module has been moved here yet: the current Aegis code is still specific to
the Aegis runtime and its research workflows. Promote code to `common/` only
after a second sandbox has a concrete dependency on it.
