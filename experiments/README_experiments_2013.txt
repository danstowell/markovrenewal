HOW WE RAN THE EXPERIMENTS IN THE STOWELL & PLUMBLEY (2013) JMLR PAPER
======================================================================

Before running any of these, you will need to have installed the "markovrenewal" code - 
see the comment in the top of "setup.py" for how to install.

Code was run on Ubuntu Studio 11.10, but will probably run on any POSIX system such as any Linux or Mac. 
The commands given below are intended to be run from a terminal, from within the "experiments" subfolder.
 - For the synthetic experiments, output data will go to "experiments/output", and PDF figures to "experiments/output/pdf".
 - For the birdsong experiment, output goes instead into the folder alongside the birdsong audio.


SYNTHETIC EXPERIMENT I: MMRP-GENERATED DATA
-------------------------------------------

Figure 5: Performance of the full and greedy inference algorithms with varying SNR.
Figure 6: Sensitivity of inference to misspecified parameters.
Figure 7: Sensitivity of inference to missed data and correlated noise.
Figure 8: Algorithm run-time for the correlated-noise test of Figure 7.

Warning: ynthetictest.py takes a long time to run.

  python ynthetictest.py
  python plotynth.py


SYNTHETIC EXPERIMENT II: AUDITORY STREAMING
-------------------------------------------

Figure 9: Examples of sequences generated.

  python -c "from abagen import *; plotabagen()"

Figure 10: MRP transition probability densities for the two synthetic models.

  python -c "from linloggmm_aba import *; aba_coherent().plot('output/pdf/plot_coherent')"
  python -c "from linloggmm_aba import *; aba_segregated().plot('output/pdf/plot_segregated')"

Figure 11: Results of generating observations under the locked, coherent or segregated
model, and then analysing them using the coherent model or the segregated model.

  python -c "from linloggmm_aba import *; plot_abagen_x_linlog()"

Figure 12: F-measure for signal/noise separation and transitions.

  python -c "from linloggmm_aba import *; multitest_abagen_linlog()"
  python plotmultitest.py


BIRDSONG AUDIO EXPERIMENT
-------------------------

The scripts here all assume data is in "~/birdsong/xenocanto_chiffchaff", but you should be able to edit the path straightforwardly.

Note that this experiment requires you to download MP3 data from Xeno-Canto (see Table I in the paper for details),
and put the MP3s in an "mp3" subfolder of the data folder.
We then run a script to convert MP3 to WAV, and a second script to trim the WAVs to a common length:

  bash chiffchaff_mp3wav.bash
  python chiffchaff_trim.py

Figure 15: The evaluation measures for the Chiffchaff audio analyses.
Figure 16: As Figure 15, with ideal-recovery results superimposed.

  python chiffchaff.py


