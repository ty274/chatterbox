## List of commands

Generate audio files: 

    python ./tts_batch.py --transcripts ./data/text/abbr/ct_transcript.txt --outputdir ./data/audio --audio_prompts AUDIO_CATALOG_FILE.json
    python ./tts_batch.py --transcripts ./data/text/abbr/la_transcript.txt --outputdir ./data/audio2 --audio_prompts AUDIO_CATALOG_FILE.json --seed 100    

AUDIO_CATLOG_FILE.json is structured as follows: 

    [
        {
            "filename": "abc.flac",
            "speaker_id": "1089",
            "session_id": "134686",
            "length": 9635,
            "length_readable": "00:00:09.635"
        },
        {
            "filename": "xyz.flac",
            "speaker_id": "1089",
            "session_id": "134686",
            "length": 4405,
            "length_readable": "00:00:04.405"
        },
        ...
    ]
