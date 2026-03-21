"""Tool 1: Query AMP databases for known peptides against a target pathogen."""

import pandas as pd
import numpy as np
from tools import BaseTool, ToolResult

# Curated antifungal AMPs from literature — fallback if modlamp/DRAMP unavailable
CURATED_ANTIFUNGAL_AMPS = [
    # (name, sequence, source_organism, target, MIC_uM_approx)
    # Validated anti-Botrytis AMPs (J. Agric. Food Chem., ScienceDirect 2022)
    ("Epinecidin-1", "GFIFHIIKGLFHAGKMIHGLV", "Epinephelus coioides", "Botrytis cinerea", 12.5),
    ("EPI-4", "GFIFHIIKGLFHAGKMIHGLVK", "Synthetic (Epinecidin-1 variant)", "Botrytis cinerea", 6.0),
    ("Magainin 2", "GIGKFLHSAKKFGKAFVGEIMNS", "Xenopus laevis", "Broad-spectrum antifungal", 50),
    ("Cecropin B", "KWKVFKKIEKMGRNIRNGIVKAGPAIAVLGEAKAL", "Hyalophora cecropia", "Botrytis cinerea", 25),
    ("Dermaseptin S1", "ALWKTMLKKLGTMALHAGKAALGAAADTISQTQ", "Phyllomedusa sauvagii", "Candida, Aspergillus", 10),
    ("Thanatin", "GSKKPVPIIYCNRRTGKCQRM", "Podisus maculiventris", "Fusarium", 15),
    ("Indolicidin", "ILPWKWPWWPWRR", "Bos taurus", "Antifungal broad", 32),
    ("Protegrin-1", "RGGRLCYCRRRFCVCVGR", "Sus scrofa", "Antifungal broad", 8),
    ("Histatin 5", "DSHAKRHHGYKRKFHEKHHSHRGY", "Homo sapiens", "Candida albicans", 15),
    ("Lactoferricin B", "FKCRRWQWRMKKLGAPSITCVRRAF", "Bos taurus", "Antifungal broad", 20),
    ("Psd1", "KTCENLADTFRGPCFATSNCTYTECGFTGKCRDDADKCASY", "Pisum sativum", "Fusarium solani", 5),
    ("NaD1", "RECKTESNTFPGICITKPPCRKACISEKFTDGHCSKILRRCLCTKPC", "Nicotiana alata", "Botrytis, Fusarium", 2),
    ("Rs-AFP2", "QKLCERPSGTWSGVCGNNNACKNQCIRLEKARHGSCNYVFPAHKCICYFPC", "Raphanus sativus", "Botrytis cinerea", 3),
    ("MsDef1", "RTCENLANTYRGPCFTTGSCDDHCKNKEHLLSGRCRDDFCNKLRC", "Medicago sativa", "Fusarium graminearum", 4),
    ("Aurein 1.2", "GLFDIIKKIAESF", "Litoria aurea", "Antifungal", 50),
    ("Temporin A", "FLPLIGRVLSGIL", "Rana temporaria", "Antifungal", 25),
    ("Melittin", "GIGAVLKVLTTGLPALISWIKRKRQQ", "Apis mellifera", "Broad-spectrum", 5),
    ("LL-37", "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES", "Homo sapiens", "Antifungal", 15),
    ("Polymyxin B nonapeptide", "KWDKFWKNR", "Synthetic", "Gram-negative", 50),
    ("Mastoparan", "INLKALAALAKKIL", "Vespula lewisii", "Antifungal", 30),
    ("Plectasin", "GFGCNGPWDEDDMQCHNHCKSIKGYKGGYCARGGFVCKCY", "Pseudoplectania nigrella", "Fusarium", 8),
    ("AFP (Aspergillus)", "IATSPYYACNCIMIPGARYKQIGSCSGGTTNYCEKAILDGHRTHK", "Aspergillus giganteus", "Botrytis, Fusarium", 1),
    # Additional plant defensins relevant to crops
    ("Dm-AMP1", "ELCEKASKTFSGNCKNRCIRLEKARHGSCNYVFPAHKCICYFPC", "Dahlia merckii", "Botrytis, Fusarium", 3),
    ("Mj-AMP1", "QKCLNPASDFPGPCFTDSNCKYTECGFSGKKCRDDADRCASY", "Mirabilis jalapa", "Botrytis cinerea", 5),
    ("Ace-AMP1", "VGECVRGRCPSGMCCSQFGYCGKGPKYCGRHN", "Allium cepa", "Botrytis, Fusarium", 10),
    ("Hevein", "EQCGRQAGGKLCPNNLCCSQWGWCGSTDEYCSPDHNCQSNCKD", "Hevea brasiliensis", "Multiple fungi", 15),
    ("Osmotin", "ATFNITNCPFTVWAASVPIGQFYSQDLDLSSGQTFSCTADAKTFRDACTFSCNGVNAFPNSALVPGGTGGASLQE", "Nicotiana tabacum", "Fusarium, Phytophthora", 8),
]

# Map pathogen names to search terms for filtering
PATHOGEN_KEYWORDS = {
    "Botrytis cinerea": ["botrytis", "antifungal", "fungal", "fungi"],
    "Fusarium graminearum": ["fusarium", "antifungal", "fungal", "fungi"],
    "Xanthomonas citri": ["xanthomonas", "antibacterial", "gram-negative", "bacteria"],
    "Ralstonia solanacearum": ["ralstonia", "antibacterial", "gram-negative", "bacteria"],
}


class DataQueryTool(BaseTool):
    name = "Database Query"
    description = "Consulta bases de datos de péptidos antimicrobianos"
    icon = "🔬"

    def __init__(self):
        self._full_dataset = None
        self._amp_sequences = None
        self._non_amp_sequences = None

    def _load_modlamp_data(self):
        """Load modlAMP built-in AMP vs non-AMP dataset."""
        try:
            import numpy as np
            from modlamp.datasets import load_AMPvsUniProt
            data = load_AMPvsUniProt()
            seqs = data["sequences"]
            targets = np.array(data["target"])
            # target_names: ['UniProt', 'AMP'] → 0=non-AMP, 1=AMP
            self._amp_sequences = [seqs[i] for i in range(len(seqs)) if targets[i] == 1]
            self._non_amp_sequences = [seqs[i] for i in range(len(seqs)) if targets[i] == 0]
            return True
        except Exception:
            return False

    def _get_curated_data(self, pathogen: str) -> pd.DataFrame:
        """Get curated AMPs filtered by pathogen relevance."""
        keywords = PATHOGEN_KEYWORDS.get(pathogen, ["antifungal", "fungi"])
        rows = []
        for name, seq, source, target, mic in CURATED_ANTIFUNGAL_AMPS:
            target_lower = target.lower()
            relevance = sum(1 for kw in keywords if kw in target_lower)
            if relevance > 0 or "broad" in target_lower:
                rows.append({
                    "name": name,
                    "sequence": seq,
                    "source_organism": source,
                    "target_activity": target,
                    "MIC_uM": mic,
                    "relevance_score": relevance,
                })
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("relevance_score", ascending=False).reset_index(drop=True)
        return df

    def _execute(self, pathogen: str = "Botrytis cinerea") -> ToolResult:
        # Try to load modlAMP for training data
        modlamp_loaded = self._load_modlamp_data()

        # Get curated AMPs for the specific pathogen
        curated_df = self._get_curated_data(pathogen)

        # Build training dataset
        if modlamp_loaded and self._amp_sequences:
            n_amp = len(self._amp_sequences)
            n_non = len(self._non_amp_sequences)
            training_source = f"modlAMP ({n_amp} AMPs + {n_non} non-AMPs)"
        else:
            # Fallback: use curated AMPs as positives, generate random negatives
            n_amp = len(curated_df)
            n_non = 0
            training_source = f"curated ({n_amp} AMPs)"

        pathogen_specific = len(curated_df)
        all_seed_seqs = curated_df["sequence"].tolist() if len(curated_df) > 0 else []

        msg = (
            f"Consulté la base de datos de péptidos antimicrobianos. "
            f"Fuente de entrenamiento: {training_source}. "
            f"Encontré {pathogen_specific} AMPs con actividad relevante contra {pathogen}. "
            f"Seleccioné los top {min(10, pathogen_specific)} como semillas para optimización."
        )

        return ToolResult(
            status="success",
            message=msg,
            data={
                "pathogen": pathogen,
                "curated_amps": curated_df.to_dict("records") if len(curated_df) > 0 else [],
                "seed_sequences": all_seed_seqs[:10],
                "all_sequences": all_seed_seqs,
                "amp_training": self._amp_sequences if self._amp_sequences else [r["sequence"] for r in curated_df.to_dict("records")],
                "non_amp_training": self._non_amp_sequences if self._non_amp_sequences else [],
                "n_pathogen_specific": pathogen_specific,
                "training_source": training_source,
            },
        )
