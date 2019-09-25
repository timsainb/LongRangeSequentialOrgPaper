from childes_mi.utils.general import readfile
import xmltodict
import numpy as np
import pandas as pd
import warnings


def get_participants(participants):

    participant_df = pd.DataFrame(
        columns=["participant_id", "role", "name", "age", "birthday", "sex", "language"]
    )

    for participant in participants:
        participant = participant["participant"]
        if type(participant) == list:
            participant = participant[0]

        # ensure the dictionary is valid
        missing_elements = _check_for_expected_keys(
            participant,
            expected_keys["session/participant"],
            place="participant",
            raise_error=False,
        )
        # add nan to any missing elements
        for element in missing_elements:
            participant[element] = np.nan
        participant_df.loc[len(participant_df)] = [
            participant["@id"],
            participant["role"],
            participant["name"],
            participant["age"],
            participant["birthday"],
            participant["sex"],
            participant["language"],
        ]
    return participant_df


def get_transcript_metadata(transcript_dict, xml_name):
    """ retreive transcript-level metadata from transcript files
    """
    ### get trranscript metadata
    transcript_xmlns = transcript_dict["session"]["@xmlns"]
    if "@id" in transcript_dict["session"].keys():
        transcript_id = transcript_dict["session"]["@id"]
    else:
        transcript_id = xml_name
    transcript_corpus = transcript_dict["session"]["@corpus"]
    transcript_version = transcript_dict["session"]["@version"]
    ### get header information
    transcript_date = transcript_dict["session"]["header"]["date"]
    transcript_language = transcript_dict["session"]["header"]["language"]
    if "media" in transcript_dict["session"]["header"].keys():
        transcript_media = transcript_dict["session"]["header"]["media"]
    else:
        transcript_media = np.nan
    return (
        transcript_xmlns,
        transcript_id,
        transcript_corpus,
        transcript_version,
        transcript_date,
        transcript_language,
        transcript_media,
    )


expected_keys = {
    "session": [
        "@xmlns",
        "@id",
        "@corpus",
        "@version",
        "header",
        "participants",
        "transcribers",
        "userTiers",
        "tierOrder",
        "transcript",
    ],
    "session/participant": [
        "role",
        "name",
        "@id",
        "sex",
        "birthday",
        "age",
        "language",
    ],
    "session/header": ["date", "language", "media"],
    "session/tierOrder": ["tier"],
    "session/transcript": ["u", "comment"],
    # element-wise expectations
    "sessions/transcript/u/record/": [
        "@speaker",
        "@id",
        "@excludeFromSearches",
        "orthography",
        "ipaTier",
        "alignment",
        "segment",
    ],
    "session/transcript/u/orthography": ["g"],
}


def _check_for_expected_keys(_dict, expected_keys, place="session", raise_error=False):
    """ ensures that all of the expected keys are in place
    """
    missing_elements = []
    for elem in expected_keys:
        if elem not in _dict.keys():
            if raise_error:
                raise ValueError("{} not in {}".format(elem, place))
            else:
                warnings.warn("{} not in {}".format(elem, place))
                missing_elements.append(elem)

    # check for unexpected elements
    for elem in _dict.keys():
        if elem not in expected_keys:
            warnings.warn("{} not an expected element in {}".format(elem, place))
    return missing_elements


def process_record(record_num, record):

    _check_for_expected_keys(
        record, expected_keys["sessions/transcript/u/record/"], place="record"
    )

    # check record
    _check_for_expected_keys(
        record, expected_keys["sessions/transcript/u/record/"], place="record"
    )
    # record info
    record_id = record["@id"]

    if "@speaker" not in record.keys():
        return
    record_speaker = record["@speaker"]

    # orthography
    orth = record["orthography"]["g"]
    if type(orth) is not list:
        orth = [orth]
    record_orthography = [
        words["w"] if "w" in words.keys() else np.nan for words in orth
    ]

    # timing
    if record["segment"]["@unitType"] != "ms":
        raise ValueError(
            'Segment type {} is not "ms"'.format(record["segment"]["@unitType"])
        )
    start_time = record["segment"]["@startTime"]
    duration = record["segment"]["@duration"]

    # IPA transcription
    if "ipaTier" not in record.keys():
        return
    ipaTier = record["ipaTier"]
    if type(ipaTier) is not list:
        ipaTier = [ipaTier]

    # loop through tiers
    ipaTiers = {}
    for tier in ipaTier:
        form = tier["@form"]
        phones = tier["pg"]
        if type(phones) is not list:
            phones = [phones]
        phones = [pl["w"] for pl in phones if "w" in pl.keys()]
        ipaTiers[form] = phones

    if "model" not in ipaTiers.keys():
        ipaTiers["model"] = np.nan
    if "actual" not in ipaTiers.keys():
        ipaTiers["actual"] = np.nan

    record_df = pd.DataFrame(
        [
            [
                record_id,
                record_speaker,
                record_orthography,
                start_time,
                duration,
                ipaTiers["model"],
                ipaTiers["actual"],
            ]
        ],
        columns=[
            "record_id",
            "speaker",
            "orthography",
            "start_time_ms",
            "duration_ms",
            "ipa_model",
            "ipa_actual",
        ],
    )

    return record_df


def process_transcript(xml_loc, participant_save_loc=None, transcript_save_loc=None):
    """ Processes an XML transcript into pandas dataframes
    
    Read an XML transcript and output a pandas dataframe of that 
    transcript and speakers in that transcript, returns a dataframe with 
    transcript metadata
    """
    # read file
    xml_string = readfile(xml_loc)
    # print(xml_loc.as_posix()[-50:])

    # convert xml to dict
    transcript_dict = xmltodict.parse(xml_string)
    if "session" not in transcript_dict.keys():
        warnings.warn("No transcripts for {}".format(xml_loc))
        return
    # check session
    _check_for_expected_keys(
        transcript_dict["session"], expected_keys["session"], place="session"
    )

    # check header
    _check_for_expected_keys(
        transcript_dict["session"]["header"],
        expected_keys["session/header"],
        place="session/header",
    )

    transcript_root_name = "_".join(xml_loc.as_posix()[:-4].split("/")[-5:])

    # get metadata
    (
        transcript_xmlns,
        transcript_id,
        transcript_corpus,
        transcript_version,
        transcript_date,
        transcript_language,
        transcript_media,
    ) = get_transcript_metadata(transcript_dict, xml_loc.name)

    # get a list of participants
    participants = transcript_dict["session"]["participants"]
    # ensure > 1 participant
    if type(participants) is not list:
        participants = [participants]
    participant_df = get_participants(participants)
    participant_df["transcript_id"] = transcript_id
    participant_df["xml_loc"] = xml_loc

    # get records from transcript
    records = transcript_dict["session"]["transcript"]["u"]
    if type(records) != list:
        records = [records]

    # create transcripts from records
    transcript_list = [
        process_record(record_num, record) for record_num, record in enumerate(records)
    ]
    if np.all([i is None for i in transcript_list]):
        warnings.warn("No transcripts for {}".format(xml_loc))
        return
    transcript_df = pd.concat(transcript_list)
    transcript_df["transcript_id"] = transcript_id
    transcript_df["xml_loc"] = xml_loc

    if participant_save_loc is not None:
        participant_df.to_pickle(participant_save_loc)

    if transcript_save_loc is not None:
        transcript_df.to_pickle(transcript_save_loc)

    transcript_info_df = pd.DataFrame(
        [
            [
                transcript_id,
                transcript_root_name,
                transcript_corpus,
                transcript_version,
                transcript_date,
                transcript_language,
                transcript_media,
                xml_loc,
            ]
        ],
        columns=[
            "transcript_id",
            "transcript_root_name",
            "corpus",
            "version",
            "date",
            "language",
            "media",
            "xml_loc",
        ],
    )
    return transcript_info_df
