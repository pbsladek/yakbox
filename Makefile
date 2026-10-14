.DEFAULT_GOAL := help

UV ?= uv
GH ?= gh
RELEASE_REPO ?= pbsladek/yakbox
RELEASE_REF ?= main
RELEASE_TAG ?=
RUN_ID ?=

# Pass settings as environment values so recipes preserve spaces and punctuation.
export RELEASE_REPO RELEASE_REF RELEASE_TAG RUN_ID

.PHONY: help release release-preflight release-runs release-watch

help:
	@printf '%s\n' \
	  'make release             Prepare, test, merge, and publish with Google Release Please.' \
	  'make release-preflight   Check and build the local version from a clean worktree.' \
	  'make release-runs        List release workflow runs.' \
	  'make release-watch RUN_ID=12345  Watch one run and return its exit status.' \
	  '' \
	  'Settings: RELEASE_REPO=pbsladek/yakbox RELEASE_REF=main RELEASE_TAG=' \
	  '          UV=uv GH=gh' \
	  'Every release publishes the same tested files to GitHub and PyPI.'

release:
	@if [ -z "$$RELEASE_TAG" ]; then \
	  $(UV) run python -m yakbox.release_automation --repo "$$RELEASE_REPO" \
	    --ref "$$RELEASE_REF" --gh "$(GH)"; \
	else \
	  $(GH) workflow run release.yml --repo "$$RELEASE_REPO" --ref "$$RELEASE_REF" \
	    -f "tag=$$RELEASE_TAG"; \
	fi

release-preflight:
	@set -eu; \
	tag="$$RELEASE_TAG"; \
	if [ -z "$$tag" ]; then \
	  version="$$($(UV) version --short)"; \
	  tag="v$$version"; \
	fi; \
	$(UV) run yakbox-release-preflight --tag "$$tag"

release-runs:
	$(GH) run list --repo "$$RELEASE_REPO" --workflow release.yml

release-watch:
	@case "$$RUN_ID" in \
	  ''|*[!0-9]*) printf '%s\n' 'Set RUN_ID to the numeric ID from make release-runs.' >&2; exit 2 ;; \
	esac
	$(GH) run watch "$$RUN_ID" --repo "$$RELEASE_REPO" --exit-status
