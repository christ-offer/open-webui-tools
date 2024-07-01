# Open WebUI Tools

This repository contains a collection of tools that can be used with Open WebUI.

These tools are written in python and can be easily integrated into your own WebUI.

## Tools

### Stock Reporter

A comprehensive stock analysis tool that gathers data from Finnhub API and compiles a detailed report.

#### Usage

To use this tool, you need to provide a Finnhub API key in the `Valves` section of your WebUI settings.
You also need to install the `finnhub-python` package using pip.

If you are running owui through docker, you can install it using the following command:
docker exec -it owui bash -c "pip install finnhub-python"
