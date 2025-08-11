docker := require("docker")
rm := require("rm")
uv := require("uv")


PACKAGE := "fraudsys"
REPOSITORY := "fraudsys"
SOURCES := "src"
TESTS := "tests"

default:
  @just --list

import "tasks/format.just"
