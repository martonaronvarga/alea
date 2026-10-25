# Third-party notices

Alea's package license covers only Alea-owned work. Upstream code, adapted test
fixtures, datasets and external implementations retain their own terms. Listing
an upstream project here is repository-wide attribution; it does not mean that
every Alea package bundles or links that project.

The default production closure of alea-core/math/mcmc/runtime/autodiff/smc excludes
cubature. The alea-ffi/distributions/cli package trees use GPL-3.0-or-later as
combined distributions; cubature itself retains GPL-2.0-or-later. Development,
profiling and optional backend closures need their own dependency notices.

## Cubature (native boundary)

Source: https://github.com/stevengj/cubature
Pinned revision: 4ba70a7d595a00cfb15b02763013e0088013adb7
Files: alea-ffi/vendor/cubature; local matching C fragments in cubature_patch.rs
and the allocation-failure harness. The build applies checked local safety patches;
the generated source retains the original upstream notice.
License: GPL-2.0-or-later. Original notice follows; the complete GPL text accompanies
the native package as LICENSE, and upstream COPYING.md remains in the vendored tree.

# Copyright and License

Copyright &copy; 2005-2013 Steven G. Johnson.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; **either version 2 of the License, or
(at your option) any later version.**

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Portions of hcubature C routines (see comments) based on
[HIntLib](http://www.cosy.sbg.ac.at/~rschuer/hintlib/) (also
distributed under the GNU GPL, v2 or later), copyright &copy; 2002-2005
Rudolf Schuerer.

Portions of hcubature C routines (see comments) based on [GNU
GSL](http://www.gnu.org/software/gsl/) (also distributed under the GNU
GPL, v2 or later), copyright &copy; 1996-2000 Brian Gough.


## Stan Math (adapted Wiener test fixtures)

The fixtures identified as adapted from Stan in alea-distributions/src/wiener/tests.rs
retain the following BSD-3-Clause notice. Preserved Wiener experiments in archive/
also identify Stan comparisons/adaptations. The numerical implementations and
reference comparisons are not claims of upstream endorsement.
Source license: https://github.com/stan-dev/math/blob/v5.2.0/LICENSE.md

BSD 3-Clause License

Copyright (c) 2011-2020, Stan Developers and their Assignees
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
## PosteriorDB (benchmark models, data and reference artifacts)

Pinned revision: 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
Source: https://github.com/stan-dev/posteriordb
Scope: tools/bench/corpus/vendor and tools/bench/corpus/pdb-v2, including their
source metadata and file hashes. The copied upstream LICENCE.md is retained under
corpus/vendor; individual source metadata remains authoritative.

BSD 3-Clause License

Copyright (c) 2011-2024, Stan Developers and their Assignees
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## nuts-rs (upstream reference experiments)

Source: https://github.com/pymc-devs/nuts-rs
Scope: independently fetched reference implementations and benchmark/reference
fixtures. The retained upstream license is reproduced below; this notice does not
claim that nuts-rs is a production dependency of Alea's inference crates.

MIT License

Copyright (c) 2022 Adrian Seyboldt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## BridgeStan (benchmark target backend)

Source: https://github.com/roualdes/bridgestan
Scope: the benchmark-only target backend and retained upstream licensing material.
Code is BSD-3-Clause. Upstream documentation has separate CC-BY-4.0 terms retained
in tools/bench/corpus/pdb-v2/BRIDGESTAN-LICENSE-DOC.

BSD 3-Clause License

Copyright (c) 2022, BridgeStan Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Other dependencies and prior artifacts

Registry dependencies, compiler components, Stan, JAX/BlackJAX, Enzyme, OpenBLAS and
other optional tools retain their upstream licenses. Dependency source fetched by
Cargo/Nix includes those notices; it is not relicensed by the Alea manifest. Binary
distributors must supply notices/source required by the actual linked closure.
In particular, the development-only pprof flamegraph path includes CDDL-1.0 inferno;
prepared Alea releases do not ship those profiling executables.

Historical benchmark archives preserve their measured source bytes and metadata,
including earlier manifests without license fields. They are not retroactively
changed. This notice supplies the applicable current Alea grant and upstream
attributions without rewriting those historical artifacts.

## ATLAS and PSIS references

The full ATLAS workflow and proposal-density review consult Chirag Modi's
AtlasSampler at a03c0608984a7720102230ffc19465af8868160b (MIT),
https://github.com/modichirag/AtlasSampler, and arXiv:2410.21587v1. Alea uses
the paper's Appendix E.2 density equation, deterministic HVP curvature, bounded
work and typed errors; it does not claim byte-for-byte Python equivalence.

MIT License

Copyright (c) 2024 Chirag Modi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

PSIS fitting follows Zhang and Stephens (2009) and the Apache-2.0 Stan2.38
reference policy (https://github.com/stan-dev/stan). Executed fixtures call the
upstream implementation. Stan code and model binaries used only by benchmark
helpers retain their Apache-2.0 license; they are not incorporated into the
production crate dependency graph. See LICENSE-APACHE for the license text.
