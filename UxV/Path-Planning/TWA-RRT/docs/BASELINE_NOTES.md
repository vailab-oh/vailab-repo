# Baseline provenance and implementation notes

References follow the supplied manuscript: TB [18], ST [25], STL [27], HPO [28].
The current selected files have no verified upstream source/license notices. A paper citation is not proof of code origin or permission to redistribute. No baseline is assigned MIT by this preparation.

## TB-RRT* benchmark implementation (paper label: TB-RRT)

Sintov, A., and Shapiro, A. (2014). Time-based RRT Algorithm for Rendezvous Planning of Two Dynamic Systems. ICRA, 6745-6750. [Paper](https://doi.org/10.1109/ICRA.2014.6907855).

Original-author code reuse and independent reimplementation status: unverified. Benchmark changes: star-style parent choice/rewiring; static 2D circles; fixed goal/window; common JSON and metrics. The cited paper is TB-RRT, not a verified TB-RRT* reference.

## ST-RRT* benchmark implementation

Grothe, F., Hartmann, V. N., Orthey, A., and Toussaint, M. (2022). ST-RRT*: Asymptotically-Optimal Bidirectional Motion Planning through Space-Time. ICRA. [Paper](https://doi.org/10.1109/ICRA46639.2022.9811814).

Original-author code reuse and independent reimplementation status: unverified. Benchmark changes: static 2D circles, single tree instead of original bidirectional planner; constant-speed final evaluation; common JSON and metrics.

## STL-RRT* benchmark implementation

Linard, A., Torre, I., Bartoli, E., Sleat, A., Leite, I., and Tumova, J. (2023). Real-Time RRT* with Signal Temporal Logic Preferences. IROS, 8621-8627. [Paper](https://doi.org/10.1109/IROS55552.2023.10341993).

Original-author code reuse and independent reimplementation status: unverified. Benchmark changes: static 2D planning and arrival-window cost; fixed speed; common JSON, goal tolerance, metrics and failure fallback. This is not a verified copy of the social-navigation author implementation.

## HPO-RRT* benchmark implementation

Guo, Y., Liu, X., Jia, Q., Liu, X., and Zhang, W. (2023). HPO-RRT*: A Sampling-Based Algorithm for UAV Real-Time Path Planning in a Dynamic Environment. Complex & Intelligent Systems 9, 7133-7153. [Paper](https://doi.org/10.1007/s40747-023-01115-2).

Local implementation notes describe a paper-based reimplementation; no original-author code repository is recorded. Independent authorship still requires confirmation. Benchmark changes: static 2D environment; common 6000-draw cap, speed, step, radius and evaluation; native geometric-goal termination retained. See docs/HPO_IMPLEMENTATION_NOTES.md.

## Upstream checks

- ST original-author project: https://vhartmann.com/strrt/ links OMPL. OMPL has a BSD license (https://github.com/ompl/ompl/blob/main/LICENSE). This does not establish that this local Python implementation derives from OMPL or inherits that license.
- STL author repository: https://github.com/KTH-RPL-Planiacs/STL-RT-RRTstar. A LICENSE path is listed but its actual text could not be verified in this session. No code from it was downloaded or copied. Local derivation is unverified.
- TB: no original-author code URL or license was established from local evidence.
- HPO: original-author code URL/license not recorded. The article is openly available, but article licensing does not establish the license of independently supplied code.

Before publishing, the owner must confirm whether each baseline is an independent reimplementation or contains upstream code. If upstream code was used, add its exact repository/commit, license text, copyright notices and modification record. The algorithms and their benchmark deviations were preserved, not corrected to match original papers.
