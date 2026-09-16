# Yakbox TODO

## Multi-model speech-analysis qualification

The automated contracts, worker runtimes, scheduling, caches, migration tools,
and fail-closed cutover decision are implemented. Keep the ensemble out of
public build outcomes until every qualification artifact below is complete and
the cutover report returns `ready = true`.

- [ ] Review and approve the staged transcript corpus. The current authoring
      state has 75 cases and no approvals; use the reduced source-anchored
      packet for the 46 exceptions and 10 audit cases.
- [ ] Complete both human boundary-review passes for the selected cases and
      accept the final boundary truth.
- [ ] Generate and freeze deterministic risk variants for every registered
      defect class, retaining at least 52 independent held-out clusters per
      class.
- [ ] Freeze the strict truth table, fit calibration thresholds on the
      calibration partition, and run the held-out shadow evaluation.
- [ ] Run the full-corpus Qwen BF16/8-bit and Parakeet offline/streaming
      comparisons.
- [ ] Run and approve the frozen reference-M5 performance protocol.
- [ ] Run the extended release preflight from a clean release worktree and
      preserve the worker SBOMs and other release evidence.
- [ ] Collect the final source-anchor, transcript, truth-table, and threshold
      approvals, then evaluate the cutover-readiness report.
- [ ] Make manifest v2 and the strict ensemble public only after the report
      returns `ready = true`.

See `docs/speech-analysis-qualification.md` for the evidence contracts and
operator commands.

## Local Chatterbox narration and short-utterance approval

The implementation is available as an experimental, opt-in strategy. Do not
change new-project defaults until the listening and full-chapter acceptance
gates pass.

- [ ] Complete the generated local narration scorecard with a named reviewer,
      preferred profile, approved settings, voice and chapter scores, join
      observations, and any blocking defects.
- [ ] Validate the completed `qa/listening-review.toml` with
      `tests/live/test_narration_review.py`.
- [ ] Review the one-word `Third.` and `No.` crops for extra speech, consonant
      loss, clicks, and join quality.
- [ ] Complete and approve the full Chapter 1 release review with no blocking
      short-utterance defects.
- [ ] Decide whether the evidence supports changing the default strategy; keep
      it opt-in if any acceptance gate remains unresolved.

See `docs/local-narration-qa.md` and
`docs/short-utterance-synthesis.md` for the review procedure and acceptance
gates.
