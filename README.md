# embedd

One always-on semantic index over a folder of markdown, on CPU, that every
agent on the network shares.

The usual failure mode with local assistants is that each one grows its own
retrieval: a voice assistant embeds the notes in-process, a CLI tool builds a
second index with a different model, and the autonomous agent gets none at all
because wiring a third one was too much work. The vectors are not comparable, so
the indexes can never be merged, and the same question gets three different
answers depending on which thing you asked.

This is the other approach. One model, one chunking scheme, one index, behind a
small HTTP API. Everything else is a thin client.

## Shape

    llama-server --embedding      the model, CPU only, bound to localhost
    embedd.py                     the index + search API, bound to the network
    vault-search                  the client, on every machine

The index refreshes itself on a timer (incremental, by mtime), so callers never
think about freshness. Search is a dot product against one in-memory matrix:
10k chunks at 768 dimensions is 30 MB of RAM and well under a millisecond, so
there is no vector database here and no need for one.

## API

    GET  /health                     model, dim, chunk/file counts, index age
    POST /search  {q, k, path}       top-k passages; `path` filters by prefix
    POST /embed   {input: [...]}     raw vectors, for callers with their own store
    POST /reindex {full: false}      force a pass now

## Use

    vault-search "when did the fan fail"
    vault-search -k 10 --path College/ "what is due"
    vault-search --json "..."          # for agents
    vault-search --health

## Install

Needs `llama-server` from llama.cpp, an embedding model in GGUF, python3 and
numpy. Nothing else.

    # 1. the model (768-dim, 274 MB, matches ollama's nomic-embed-text)
    curl -L -o nomic-embed-text-v1.5.f16.gguf \
      https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf

    # 2. edit paths in systemd/*.service, then
    cp systemd/*.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now vault-embed-model.service vault-embed.service

    # 3. on every other machine
    install -m755 vault-search ~/bin/
    mkdir -p ~/.config/vault-search && echo http://<host>:11475 > ~/.config/vault-search/url

Config is environment only: `EMBED_VAULT` `EMBED_STATE` `EMBED_URL` `EMBED_PORT`
`EMBED_HOST` `EMBED_INTERVAL` `EMBED_EXCLUDE` `EMBED_MODEL`.

## Notes

Dot-directories are always skipped. A Syncthing `.stversions` folder can hold ten
times more stale copies of your notes than the notes themselves, and they will
swamp every result if you let them in.

The embedding model runs on CPU deliberately. It is small, it is asked a question
a few times a minute, and the GPUs have better things to do.

MIT.
