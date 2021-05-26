from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from typing import List, Dict, Tuple, Any, Union

from deepPavlovEval.download_utils import download

log = getLogger(__name__)


class STSReader:
    """
    Data description:
        data_path is a directory with two files:
            input_fname
            labels_fname

        input_fname file structure:
            tab-separated pair of sentences on each line
        labels_fname file structure:
            floatable number on each line - sentence similarity

        where input_fname and labels_fname are .__init__() parameters
        with default values
    Example:
        input_fname:

        A woman and man are dancing in the rain.	A man and woman are dancing in rain.
        Someone is drawing.	Someone is dancing.

        labels_fname:

        5.000
        0.300

    Return:
        {'test': test_set}, where test_set is a list of tuples (sent1, sent2, similarity)
    """
    def read(self, data_path: str, input_fname='input.txt', labels_fname='labels.txt', *args, **kwargs):
        data_path = Path(data_path)

        with open(data_path / input_fname, encoding='utf-8') as f:
            data = [l.rstrip('\n').split('\t') for l in f.readlines()]

        with open(data_path / labels_fname, encoding='utf-8') as f:
            labels = [float(l.rstrip('\n')) for l in f.readlines()]

        merged = [((e[0], e[1]), l) for e, l in zip(data, labels)]
        return {'test': merged}


class XNLIReader:
    """
    Data description:
        not_implemented_error
    
    Example:
        not_implemented_error

    Return:
        {'valid': valid_set, 'test': test_set}
        where valid_set and test_set are np.arrays [((sent1, sent2), label), ...]
    """
    def read(self, data_path, valid_fname='xnli.dev.tsv', test_fname='xnli.test.tsv', lang=None):
        data_path = Path(data_path)
        valid = self.read_one(data_path / valid_fname, lang=lang)
        test = self.read_one(data_path / test_fname, lang=lang)
        return {'train': valid, 'test': test}

    def read_one(self, data_path, lang):
        data = pd.read_csv(data_path, sep='\t')
        data = data[['language', 'gold_label', 'sentence1', 'sentence2']]
        if lang is not None:
            data = data[data.language == lang]

        data = data[['sentence1', 'sentence2', 'gold_label']].values
        data = [((s[0], s[1]), s[2]) for s in data]
        return data


class DatasetReader:
    """An abstract class for reading data from some location and construction of a dataset."""

    def read(self, data_path: str, *args, **kwargs) -> Dict[str, List[Tuple[Any, Any]]]:
        """Reads a file from a path and returns data as a list of tuples of inputs and correct outputs
         for every data type in ``train``, ``valid`` and ``test``.
        """
        raise NotImplementedError


def expand_path(path: Union[str, Path]) -> Path:
    """Convert relative paths to absolute with resolving user directory."""
    return Path(path).expanduser().resolve()


class ParaphraserReader(DatasetReader):
    """The class to read the paraphraser.ru dataset from files.

    Please, see https://paraphraser.ru.
    """

    def read(self,
             data_path: str,
             do_lower_case: bool = True,
             *args, **kwargs) -> Dict[str, List[Tuple[Tuple[str, str], int]]]:
        """Read the paraphraser.ru dataset from files.

        Args:
            data_path: A path to a folder with dataset files.
            do_lower_case: Do you want to lowercase all texts
        """

        data_path = expand_path(data_path)
        train_fname = data_path / 'paraphrases.xml'
        test_fname = data_path / 'paraphrases_gold.xml'

        train_data = self._build_data(train_fname, do_lower_case)
        test_data = self._build_data(test_fname, do_lower_case)
        return {"train": train_data, "valid": [], "test": test_data}

    @staticmethod
    def _build_data(data_path: Path, do_lower_case: bool) -> List[Tuple[Tuple[str, str], int]]:
        root = ET.fromstring(data_path.read_text(encoding='utf8'))
        data = {}
        for paraphrase in root.findall('corpus/paraphrase'):
            key = (paraphrase.find('value[@name="text_1"]').text,
                   paraphrase.find('value[@name="text_2"]').text)
            if do_lower_case:
                key = tuple([t.lower() for t in key])

            data[key] = 1 if int(paraphrase.find('value[@name="class"]').text) >= 0 else 0
        return list(data.items())


class BasicClassificationDatasetReader(DatasetReader):
    """
    Class provides reading dataset in .csv format
    """

    def read(self, data_path: str, url: str = None,
             format: str = "csv", class_sep: str = None,
             *args, **kwargs) -> dict:
        """
        Read dataset from data_path directory.
        Reading files are all data_types + extension
        (i.e for data_types=["train", "valid"] files "train.csv" and "valid.csv" form
        data_path will be read)

        Args:
            data_path: directory with files
            url: download data files if data_path not exists or empty
            format: extension of files. Set of Values: ``"csv", "json"``
            class_sep: string separator of labels in column with labels
            sep (str): delimeter for ``"csv"`` files. Default: None -> only one class per sample
            header (int): row number to use as the column names
            names (array): list of column names to use
            orient (str): indication of expected JSON string format
            lines (boolean): read the file as a json object per line. Default: ``False``

        Returns:
            dictionary with types from data_types.
            Each field of dictionary is a list of tuples (x_i, y_i)
        """
        data_types = ["train", "valid", "test"]

        train_file = kwargs.get('train', 'train.csv')

        if not Path(data_path, train_file).exists():
            if url is None:
                raise Exception(
                    "data path {} does not exist or is empty, and download url parameter not specified!".format(
                        data_path))
            log.info("Loading train data from {} to {}".format(url, data_path))
            download(source_url=url, dest_file_path=Path(data_path, train_file))

        data = {"train": [],
                "valid": [],
                "test": []}
        for data_type in data_types:
            file_name = kwargs.get(data_type, '{}.{}'.format(data_type, format))
            if file_name is None:
                continue

            file = Path(data_path).joinpath(file_name)
            if file.exists():
                if format == 'csv':
                    keys = ('sep', 'header', 'names')
                    options = {k: kwargs[k] for k in keys if k in kwargs}
                    df = pd.read_csv(file, **options)
                elif format == 'json':
                    keys = ('orient', 'lines')
                    options = {k: kwargs[k] for k in keys if k in kwargs}
                    df = pd.read_json(file, **options)
                else:
                    raise Exception('Unsupported file format: {}'.format(format))

                x = kwargs.get("x", "text")
                y = kwargs.get('y', 'labels')
                if isinstance(x, list):
                    if class_sep is None:
                        # each sample is a tuple ("text", "label")
                        data[data_type] = [([row[x_] for x_ in x], str(row[y]))
                                           for _, row in df.iterrows()]
                    else:
                        # each sample is a tuple ("text", ["label", "label", ...])
                        data[data_type] = [([row[x_] for x_ in x], str(row[y]).split(class_sep))
                                           for _, row in df.iterrows()]
                else:
                    if class_sep is None:
                        # each sample is a tuple ("text", "label")
                        data[data_type] = [(row[x], str(row[y])) for _, row in df.iterrows()]
                    else:
                        # each sample is a tuple ("text", ["label", "label", ...])
                        data[data_type] = [(row[x], str(row[y]).split(class_sep)) for _, row in df.iterrows()]
            else:
                log.warning("Cannot find {} file".format(file))

        return data
