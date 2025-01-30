from pyannote.core import Annotation, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionCostFunction, DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.metrics.segmentation import SegmentationPurityCoverageFMeasure
from pympi.Elan import Eaf

EPISODE_ID = "384359e2-ace7-4a92-8c6d-ba07ccf41316"
ELAN_ANNOTATION = f"{EPISODE_ID}.eaf"
REFERENCE_FILE = f"R_{EPISODE_ID}.rttm"
HYPOTHESIS_FILE = f"H_{EPISODE_ID}.rttm"


def eaf_to_rttm(eaf_file, rttm_file):
    # Load the ELAN .eaf file
    eaf = Eaf(eaf_file)

    with open(rttm_file, "w") as out_file:
        output = {}
        for tier_name, annotations in eaf.tiers.items():
            if len(annotations[0]) == 0:
                continue
            # Get speaker ID from tier name
            # speaker_id = tier_name  # instead of real name, use SPEAKER_X name
            speaker_id = annotations[2]["PARTICIPANT"]
            print(tier_name + ": " + speaker_id)

            for a_id, annotation in annotations[0].items():
                start_time = eaf.timeslots[annotation[0]] / 1000.0  # Convert ms to seconds
                end_time = eaf.timeslots[annotation[1]] / 1000.0
                duration = end_time - start_time

                # RTTM fields
                rttm_line = f"SPEAKER {EPISODE_ID} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
                output[float(f"{start_time:.3f}")] = rttm_line

        sorted_keys = sorted(output.keys())
        for index in sorted_keys:
            out_file.write(output[index])


def calculate_diarization_error():
    """
    See pyannote documentation https://pyannote.github.io/pyannote-metrics/reference.html
    :return:
    """
    # Load RTTM files as pyannote Annotations
    reference = load_rttm(REFERENCE_FILE)
    hypothesis = load_rttm(HYPOTHESIS_FILE)

    # Initialize DER metric
    der = DiarizationErrorRate()
    dcf = DetectionCostFunction()
    deer = DetectionErrorRate()
    sc = SegmentationPurityCoverageFMeasure()
    ier = IdentificationErrorRate()

    # Compute DER for file in the reference
    for file_id in reference:
        ref_annotation = reference[file_id]
        hyp_annotation = hypothesis.get(file_id, Annotation())

        # Getting length of timeline
        timeline = Timeline(segments=ref_annotation._tracks)

        # Calculate DER
        file_der = der(ref_annotation, hyp_annotation, uem=timeline, detailed=True)
        file_dcf = dcf(ref_annotation, hyp_annotation, uem=timeline, detailed=True)
        file_deer = deer(ref_annotation, hyp_annotation, uem=timeline, detailed=True)
        file_sc = sc(ref_annotation, hyp_annotation, uem=timeline, detailed=True)
        file_ier = ier(ref_annotation, hyp_annotation, uem=timeline, detailed=True)

        print(f"DER for {file_id}")
        for key, value in file_der.items():
            print(f"{key}: {value:.4f}")

        print(f"\nDCF for {file_id}")
        for key, value in file_dcf.items():
            print(f"{key}: {value:.4f}")

        print(f"\nDER for {file_id}")
        for key, value in file_deer.items():
            print(f"{key}: {value:.4f}")

        print(f"\nSC for {file_id}")
        for key, value in file_sc.items():
            print(f"{key}: {value:.4f}")

        print(f"\nIER for {file_id}")
        for key, value in file_ier.items():
            print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    # eaf_to_rttm(ELAN_ANNOTATION, REFERENCE_FILE)
    calculate_diarization_error()
