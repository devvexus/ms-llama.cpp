# Releasing ms-llama.cpp

Like upstream llama.cpp: every push to `master` makes a pre-release with binaries; a full release is a pre-release you promote.

## What happens on its own

A push to `master` that changes code runs **MS Release** (`.github/workflows/ms-release.yml`, GitHub-hosted runners only).
It builds and publishes a **pre-release** tagged `ms-b<N>`, where `N` is the number of commits on `master`
(`git rev-list --count master`). The `ms-` prefix keeps it apart from upstream's `b<N>` tags.

Files of every release (the names never change, so `releases/latest/download/<name>` always works):

| File | Contents |
|---|---|
| `ms-llama-win-x64-cuda12.zip` | Windows x64: tools, server, CPU backend, CUDA 12.4 backend and DLLs (README install line) |
| `ms-llama-win-arm64-cuda.zip` | Windows ARM64: the same with CUDA 13.5 (README install line) |
| `ms-llama-win-x64-cpu.zip`, `ms-llama-win-arm64-cpu.zip` | Windows, CPU only |
| `ms-llama-linux-x64-cpu.tar.gz`, `ms-llama-linux-arm64-cpu.tar.gz` | Linux, CPU only |
| `ms-llama-linux-x64-cuda13.tar.gz`, `ms-llama-linux-arm64-cuda13.tar.gz` | Linux with CUDA 13.5 and its runtime libraries |
| `ms-llama-macos-arm64.tar.gz`, `ms-llama-macos-x64.tar.gz` | macOS (Metal on arm64) |

Put `[no release]` in a commit message to skip the pre-release of that push.

## Cut a release

Pick a pre-release you have tested (Releases page, or `gh release list -R devvexus/ms-llama.cpp`), then:

```
gh workflow run ms-promote.yml -R devvexus/ms-llama.cpp -f version=ms-v1.1 -f from=ms-b12345
```

Leave out `-f from=...` to promote the newest pre-release. This creates the release `ms-v1.1` at the same commit with the
same files and marks it **latest** (the README install line then downloads it). No new build.

In the browser: **Actions > MS Promote > Run workflow**, fill in `version` (and `from`), **Run workflow**.

To build the current `master` and release it in one go (about an hour):

```
gh workflow run ms-release.yml -R devvexus/ms-llama.cpp --ref master -f version=ms-v1.1
```

Follow a run with `gh run watch -R devvexus/ms-llama.cpp`; check the result with `gh release view ms-v1.1 -R devvexus/ms-llama.cpp`.

## Rules

- Version tags are `ms-vX.Y` or `ms-vX.Y.Z`; the workflows refuse other names and an existing tag.
- Every release reads both MS file generations: Gen 1 (type ids 43..48, the files made for `ms-v1.0`) is mapped to the Gen 2
  ids (201..206) at load time; new MS files are written with the Gen 2 ids. Nothing to do for users.
- Upstream's own release workflows (`release.yml`, `release-publish.yml`, `make-release.yml`) only run in ggml-org/llama.cpp.
- To withdraw a release: `gh release delete ms-v1.1 -R devvexus/ms-llama.cpp --cleanup-tag` (asks first).
- Publishing uses the repository secret `MS_RELEASE_TOKEN`: a fine-grained personal access token (owner devvexus, only this
  repository, Contents: read and write). The job's own token gets HTTP 403 on `POST /releases`. The secret was added on
  2026-11-22 with a 1-year expiry: when GitHub warns that it expires, make a new token the same way and replace the secret.
