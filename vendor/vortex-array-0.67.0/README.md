# Vortex array

TODO

Also contains

## Vortex Expressions

A crate defining serializable predicate expressions. Used predominantly for filter push-down.

Takes inspiration from postgres <https://www.postgresql.org/docs/current/sql-expressions.html>
and datafusion <https://github.com/apache/datafusion/tree/5fac581efbaffd0e6a9edf931182517524526afd/datafusion/expr>


## NeoEthos compatibility backport

This is the published vortex-array 0.67.0 package with the upstream nested
patch-slice correction from commit
[6c7eb338c17e3f938af15b43cf34c4e9e933fef6](https://github.com/vortex-data/vortex/commit/6c7eb338c17e3f938af15b43cf34c4e9e933fef6).
The parent patch base and its within-chunk offset must be restored before
subtracting the new chunk base. Otherwise a second valid slice can underflow.
The upstream regression and a dense exact-value composition oracle cover it.
The package version, other Vortex dependencies and wire format are unchanged.
Remove this override only after a compatible published upstream release passes
the same nested-slice and NeoEthos projection/receipt parity regressions.
