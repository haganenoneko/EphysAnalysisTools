<!--
 Copyright (c) 2021 Delbert Yip
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

<style>
    pre {border: 5%; font-size: 14px;}
</style>

#  Style guidelines 
These are style guidelines (some are arbitrary) for personal use. They do not necessarily conform to PEP8 or other standards, but generally do. 

## Classes 
- every word in a class name is capitalized, e.g. `WordWordWord`
  - underscores are only used if: 
    - more than 3 words in the class name
    - two or more similar classes, e.g. `Word_A`, `Word_B`
- 'private' methods and attributes are prefixed by leading underscores, e.g. `self._word`

## Architecture
### `@dataclass` for individual recordings
Each dataset consists of a CSV file and an ABF file. To hold these, we use the `AbstractRecording` `dataclass`. The `attrs` attribute is a dictionary that is intended to store arbitrary structures, such as:
- a `PdfPages` object to aggregate figures 
- leak-subtracted dataframe
- fitting results

### static methods group logically related methods
- `@staticmethod` is used abundantly to group methods that have a tight logical relationship
## Comments
- comments are sparse and/or one-liners, unless con
### Multiprocessing 
After files are selected and `AbstractRecording`s are instantiated for each selected file, 