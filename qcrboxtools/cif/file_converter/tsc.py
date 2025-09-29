import struct
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from textwrap import wrap
from typing import Dict, List, Tuple, Union

import numpy as np
from iotbx.cif.model import block, loop

from ..read import read_cif_as_unified
from .cartesian import cell_constants_to_matrix


def read_tsc_file(path: Path):
    """
    Reads a TSC or TSCB file and returns the corresponding object.
    Parameters
    ----------
    path : Path
        The path to the TSC or TSCB file.

    Returns
    -------
    TSCFile or TSCBFile
    The TSCFile or TSCBFile object representing the file content.

    Raises
    ------
    ValueError
    If the file cannot be read as either TSC or TSCB format.
    """
    path = Path(path)
    if path.suffix == ".tscb":
        try:
            return TSCBFile.from_file(path)
        except Exception as exc:
            try:
                return TSCFile.from_file(path)
            except Exception:
                raise ValueError(f"Cannot read AFF file: {str(path)}") from exc
    elif path.suffix == ".tsc":
        try:
            return TSCFile.from_file(path)
        except Exception as exc:
            try:
                return TSCBFile.from_file(path)
            except Exception:
                raise ValueError(f"Cannot read AFF file: {str(path)}") from exc


def parse_header(header_str):
    header = {}
    header_split = iter(val.split(":") for val in header_str.strip().split("\n"))

    header_key = None
    header_entry = ""
    for line_split in header_split:
        if len(line_split) == 2 and header_key is not None:
            header[header_key] = header_entry
        if len(line_split) == 2:
            header_key, header_entry = line_split
        else:
            header_entry += "\n" + line_split[0]
    header[header_key] = header_entry
    return header


def parse_tsc_data_line(line: str) -> Tuple[Tuple[int, int, int], np.ndarray]:
    """
    Parses a line of TSC data.

    Parameters
    ----------
    line : str
        The line of TSC data to parse.

    Returns
    -------
    tuple
        A tuple containing the indices h, k, l and the array of f0j values.
    """
    h_str, k_str, l_str, *f0j_strs = line.split()
    f0js = np.array([float(val.split(",")[0]) + 1j * float(val.split(",")[1]) for val in f0j_strs])
    return (int(h_str), int(k_str), int(l_str)), f0js


class TSCBase(ABC):
    def __init__(self):
        self.header = {"TITLE": "generic_tsc", "SYMM": "expanded", "SCATTERERS": ""}
        self.data = {}

    @property
    def scatterers(self) -> List[str]:
        """
        Retrieves scatterers from the TSC file as a list of strings generated
        from the SCATTERERS header entry.

        Returns
        -------
        list
            A list of scatterer names.
        """

        return self.header["SCATTERERS"].strip().split()

    @scatterers.setter
    def scatterers(self, scatterers: Iterable):
        """
        Sets the scatterers in the TSC file.

        The input scatterers are converted to a space-separated string and
        stored in the header under the key 'SCATTERERS'.

        Parameters
        ----------
        scatterers : iterable
            An iterable of scatterer names.
        """
        self.header["SCATTERERS"] = " ".join(str(val) for val in scatterers)

    def __getitem__(self, atom_site_label: Union[str, Iterable]) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Retrieves f0j values for a given atom site label.

        The function allows indexing the TSCFile object by atom site label or a
        list of labels. If the given label is not found among the scatterers,
        a ValueError is raised.

        Parameters
        ----------
        atom_site_label : str or iterable
            The atom site label or a list of labels to retrieve f0j values for.

        Returns
        -------
        dict
            A dictionary where each key is a tuple of indices (h, k, l) and the
            corresponding value is a numpy array of f0j values for the given
            label(s).

        Raises
        ------
        ValueError
            If an unknown atom site label is used for indexing.
        """
        try:
            if isinstance(atom_site_label, Iterable) and not isinstance(atom_site_label, str):
                indexes = np.array([self.scatterers.index(label) for label in atom_site_label])
                return {hkl: f0js[indexes] for hkl, f0js in self.data.items()}
            else:
                index = self.scatterers.index(atom_site_label)
                return {hkl: f0js[index] for hkl, f0js in self.data.items()}
        except ValueError as exc:
            if isinstance(atom_site_label, Iterable) and not isinstance(atom_site_label, str):
                unknown = [label for label in atom_site_label if label not in self.scatterers]
            else:
                unknown = [atom_site_label]
            raise ValueError(f"Unknown atom label(s) used for lookup from TSCFile: {' '.join(unknown)}") from exc

    @classmethod
    @abstractmethod
    def from_file(cls, filename: Path):
        pass

    @abstractmethod
    def to_file(self, filename: Path):
        pass

    def _construct_moiety_loop(self, structure_cif_block: block):
        """
        Constructs a CIF loop containing moiety information from a given CIF block.

        Parameters
        ----------
        structure_cif_block : block
            The CIF block containing the structure information.

        Returns
        -------
        loop
            A CIF loop containing the moiety information.
        """
        cell_a = float(structure_cif_block["_cell.length_a"])
        cell_b = float(structure_cif_block["_cell.length_b"])
        cell_c = float(structure_cif_block["_cell.length_c"])
        alpha = float(structure_cif_block["_cell.angle_alpha"])
        beta = float(structure_cif_block["_cell.angle_beta"])
        gamma = float(structure_cif_block["_cell.angle_gamma"])
        cell_mat_m = cell_constants_to_matrix(cell_a, cell_b, cell_c, alpha, beta, gamma)
        fract_x = np.array([float(val) for val in structure_cif_block["_atom_site.fract_x"]])
        fract_y = np.array([float(val) for val in structure_cif_block["_atom_site.fract_y"]])
        fract_z = np.array([float(val) for val in structure_cif_block["_atom_site.fract_z"]])

        xyz_fract = np.stack((fract_x, fract_y, fract_z), axis=-1)
        xyz_cart = np.einsum("xy, zy -> zx", cell_mat_m, xyz_fract)
        cart_x, cart_y, cart_z = xyz_cart.T

        n_atoms = len(cart_x)

        # TODO revisit this as more sophisticated moiety handling is implemented
        moiety_loop_data = {
            "_wfn_moiety.id": np.full(n_atoms, 1),
            "_wfn_moiety.atom_id": np.arange(1, n_atoms + 1),
            "_wfn_moiety.asu_atom_site_label": structure_cif_block["_atom_site.label"],
            "_wfn_moiety.atom_type_symbol": structure_cif_block["_atom_site.type_symbol"],
            "_wfn_moiety.symm_code": ["1_555"] * n_atoms,
            "_wfn_moiety.cartn_x": list(cart_x),
            "_wfn_moiety.cartn_y": list(cart_y),
            "_wfn_moiety.cartn_z": list(cart_z),
            "_wfn_moiety.aff_index": [
                self.scatterers.index(name) + 1 for name in structure_cif_block["_atom_site.label"]
            ],
        }

        return loop(data=moiety_loop_data)

    def _construct_aff_loop(self):
        def create_aff_line_string(values):
            converted = [f"{val: 3.8f}" for val in values]
            single_line = " ".join(converted)
            return "[" + "\n".join(wrap(single_line, width=2047)) + "]"

        mil_hkl = np.asarray(list(self.data.keys()))
        all_affs = np.array(list(self.data.values()))
        aff_loop_data = {
            "_aspheric_ff.index_h": mil_hkl[:, 0].copy(),
            "_aspheric_ff.index_k": mil_hkl[:, 1].copy(),
            "_aspheric_ff.index_l": mil_hkl[:, 2].copy(),
            "_aspheric_ff.form_factor_real": list(create_aff_line_string(line) for line in np.real(all_affs)),
            "_aspheric_ff.form_factor_imag": list(create_aff_line_string(line) for line in np.imag(all_affs)),
        }
        return loop(data=aff_loop_data)

    def to_cif(
        self, structure_cif_block: block, partitioning_source: str, partitioning_name: str, partitioning_software: str
    ) -> block:
        """
        Converts the TSC data to a CIF block format.

        Parameters
        ----------
        structure_cif_block : block
            The CIF block containing the structure information.
        partitioning_source : str
            The source of the partitioning.
        partitioning_name : str
            The name of the partitioning scheme employed.
        partitioning_software : str
            The software used for the partitioning.

        Returns
        -------
        block
            A CIF block containing the TSC data.
        """
        tsc_block = block()
        tsc_block.add_data_item("_cell.length_a", structure_cif_block["_cell.length_a"])
        tsc_block.add_data_item("_cell.length_b", structure_cif_block["_cell.length_b"])
        tsc_block.add_data_item("_cell.length_c", structure_cif_block["_cell.length_c"])
        tsc_block.add_data_item("_cell.angle_alpha", structure_cif_block["_cell.angle_alpha"])
        tsc_block.add_data_item("_cell.angle_beta", structure_cif_block["_cell.angle_beta"])
        tsc_block.add_data_item("_cell.angle_gamma", structure_cif_block["_cell.angle_gamma"])
        tsc_block.add_loop(self._construct_moiety_loop(structure_cif_block))
        tsc_block.add_data_item("_aspheric_ffs.source", partitioning_source)
        tsc_block.add_data_item("_aspheric_ffs_partitioning.name", partitioning_name)
        tsc_block.add_data_item("_aspheric_ffs_partitioning.software", partitioning_software)
        tsc_block.add_loop(self._construct_aff_loop())

        return tsc_block

    def populate_from_cif_block(self, cif_block: block):
        """
        Populates the TSCFile object from a CIF block created by the TSC to cif export function.
        Parameters
        ----------
        cif_block : block
            The CIF block containing the TSC data.
        Raises
        ------
        ValueError
            If the CIF block does not contain the required entries.
        """
        if (
            "_aspheric_ffs.source" not in cif_block
            or "_aspheric_ffs_partitioning.name" not in cif_block
            or "_aspheric_ffs_partitioning.software" not in cif_block
        ):
            raise ValueError("CIF block does not contain required TSC entries.")
        self.scatterers = cif_block["_wfn_moiety.asu_atom_site_label"]
        aff_loop = cif_block.get_loop("_aspheric_ff")
        if aff_loop is None:
            raise ValueError("CIF block does not contain required TSC entries for the loop _aspheric_ff.")
        hkl_zip = zip(
            aff_loop["_aspheric_ff.index_h"], aff_loop["_aspheric_ff.index_k"], aff_loop["_aspheric_ff.index_l"]
        )
        hkl_tuples = tuple((int(mil_h), int(mil_k), int(mil_l)) for mil_h, mil_k, mil_l in hkl_zip)
        real_lines = aff_loop["_aspheric_ff.form_factor_real"]
        imag_lines = aff_loop["_aspheric_ff.form_factor_imag"]
        real_vals = np.fromiter(
            (float(val) for line in real_lines for val in line.strip("[]").split()), dtype=np.float64
        )
        imag_vals = np.fromiter(
            (float(val) for line in imag_lines for val in line.strip("[]").split()), dtype=np.float64
        )
        all_affs = real_vals + 1j * imag_vals
        n_atoms = len(self.scatterers)
        if len(all_affs) % n_atoms != 0:
            raise ValueError("Number of AFF values is not a multiple of number of scatterers.")
        all_affs = all_affs.reshape((-1, n_atoms))
        self.data = {hkl: affs for hkl, affs in zip(hkl_tuples, all_affs, strict=False)}


class TSCFile(TSCBase):
    """
    A class representing a TSC file as defined in doi:10.48550/arXiv.1911.08847

    A TSC file contains atomic form factors for a list of atoms and miller
    indicees

    You can get data for atoms for example with tsc['C1'] or tsc[['C1', 'C2']]
    currently setting is not implemented this way. All data is represented
    in the data attribute

    Attributes
    ----------
    header : dict
        A dictionary holding the header information from the TSC file.
    data : dict
        A dictionary mapping tuples (h, k, l) to numpy arrays of f0j values,
        where the ordering of the values is given by the content of the
        scatterers property / the SCATTERERS entry in the header.
    """

    @classmethod
    def from_file(cls, filename: Path) -> "TSCFile":
        """
        Constructs a TSCFile object from a file.

        The function reads the TSC file, parses its header and data sections,
        and constructs a TSCFile instance with these data.

        Parameters
        ----------
        filename : Path
            The name of the TSC file to read.

        Returns
        -------
        TSCFile
            A TSCFile instance with data loaded from the file.
        """
        with open(filename, "r") as fobj:
            tsc_content = fobj.read()
        header_str, data_str = tsc_content.split("DATA:\n")

        new_obj = cls()

        new_obj.header.update(parse_header(header_str))

        parsed_iter = iter(parse_tsc_data_line(line) for line in data_str.strip().split("\n"))

        new_obj.data = {hkl: f0js for hkl, f0js in parsed_iter}

        return new_obj

    def to_file(self, filename: Path) -> None:
        """
        Writes the TSCFile object to a file.

        The function formats the header and data sections of the TSCFile object
        and writes them to a file. Currently no safety checks are implemented
        SCATTERERS and data need to match

        Parameters
        ----------
        filename : Path
            The name of the file to write.
        """
        header_str = "\n".join(f"{key}: {value}" for key, value in self.header.items())
        data_iter = iter(
            (
                f"{int(hkl[0])} {int(hkl[1])} {int(hkl[2])} "
                + f"{' '.join(f'{np.real(val):.8e},{np.imag(val):.8e}' for val in values)}"
            )
            for hkl, values in self.data.items()
        )
        data_str = "\n".join(data_iter)

        with open(filename, "w") as fobj:
            fobj.write(f"{header_str}\nDATA:\n{data_str}\n")

    @classmethod
    def from_cif_file(cls, cif_path: Path) -> "TSCFile":
        """
        Constructs a TSCFile object from a CIF file created by the TSC to cif export function.

        Parameters
        ----------
        filename : Path
            The name of the CIF file to read.

        Returns
        -------
        TSCFile
            A TSCFile instance with data loaded from the CIF file.
        """
        cif_block = read_cif_as_unified(cif_path, 0)
        new_obj = cls()
        new_obj.populate_from_cif_block(cif_block)
        return new_obj


class TSCBFile(TSCBase):
    """
    A class representing a TSCB file used by for example NoSpherA2

    A TSC file contains atomic form factors for a list of atoms and miller
    indicees

    You can get data for atoms for example with tsc['C1'] or tsc[['C1', 'C2']]
    currently setting is not implemented this way. All data is represented
    in the data attribute

    Attributes
    ----------
    header : dict
        A dictionary holding the header information from the TSC file.
    data : dict
        A dictionary mapping tuples (h, k, l) to numpy arrays of f0j values,
        where the ordering of the values is given by the content of the
        scatterers property / the SCATTERERS entry in the header.
    """

    @classmethod
    def from_file(cls, filename: Path) -> "TSCBFile":
        """
        Constructs a TSCFile object from a file.

        The function reads the TSC file, parses its header and data sections,
        and constructs a TSCFile instance with these data.

        Parameters
        ----------
        filename : Path
            The name of the TSC file to read.

        Returns
        -------
        TSCFile
            A TSCBFile instance with data loaded from the file.
        """
        new_obj = cls()
        with open(filename, "rb") as fobj:
            additional_header_size, n_bytes_labels = struct.unpack("2i", fobj.read(8))
            if additional_header_size > 0:
                header_str = fobj.read(additional_header_size).decode("ASCII")

                new_obj.header.update(parse_header(header_str))
            new_obj.header["SCATTERERS"] = fobj.read(n_bytes_labels).decode("ASCII")

            n_refln = struct.unpack("i", fobj.read(4))[0]
            n_atoms = len(new_obj.header["SCATTERERS"].split())
            new_obj.data = {
                tuple(np.frombuffer(fobj.read(12), dtype=np.int32)): np.frombuffer(
                    fobj.read(n_atoms * 16), dtype=np.complex128
                )
                for i in range(n_refln)
            }
        return new_obj

    def to_file(self, filename: Path) -> None:
        """
        Writes the TSCBFile object to a file.

        The function formats the header and data sections of the TSCBFile object
        and writes them to a file. Currently no safety checks are implemented
        SCATTERERS and data need to match

        Parameters
        ----------
        filename : str
            The name of the file to write.
        """
        if not next(iter(self.data.values())).dtype == np.complex128:
            self.data = {key: value.astype(np.complex128) for key, value in self.data.items()}
        omitted_header_entries = ("SCATTERERS", "TITLE", "SYMM")
        header_string = "\n".join(
            f"{name}: {entry}" for name, entry in self.header.items() if name not in omitted_header_entries
        )
        with open(filename, "wb") as fobj:
            fobj.write(struct.pack("2i", len(header_string), len(self.header["SCATTERERS"])))
            fobj.write(header_string.encode("ASCII"))
            fobj.write(self.header["SCATTERERS"].encode("ASCII"))
            fobj.write(struct.pack("i", len(self.data)))
            fobj.write(bytes().join(struct.pack("3i", *hkl) + f0js.tobytes() for hkl, f0js in self.data.items()))

    @classmethod
    def from_cif_file(cls, cif_path: Path) -> "TSCBFile":
        """
        Constructs a TSCFile object from a CIF file created by the TSC to cif export function.

        Parameters
        ----------
        filename : Path
            The name of the CIF file to read.

        Returns
        -------
        TSCBFile
            A TSCBFile instance with data loaded from the CIF file.
        """
        cif_block = read_cif_as_unified(cif_path, 0)
        new_obj = cls()
        new_obj.populate_from_cif_block(cif_block)
        return new_obj
