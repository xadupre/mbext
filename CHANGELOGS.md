# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Documentation & CI

- Describes mbext as a "Fast Experimental Exporter for transformers" in package
  metadata and documentation.
- Builds a universal Python wheel and source distribution for every pushed tag
  and publishes them to the corresponding GitHub Release.
- Supports rebuilding an existing tag through a manual workflow run, replacing
  matching release assets without changing existing release notes.
