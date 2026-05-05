import asyncio
import math
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
import music21

_pitch_model = None


def load_model():
    global _pitch_model
    _pitch_model = Model(ICASSP_2022_MODEL_PATH)
    return _pitch_model


def get_loaded_model():
    if _pitch_model is None:
        load_model()
    return _pitch_model


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def _remove_harmonics(part):
    for measure in part.getElementsByClass("Measure"):
        offset_map = {}
        for note in measure.recurse().notes:
            if hasattr(note, 'pitch'):
                offset = round(float(note.offset), 3)
                if offset not in offset_map:
                    offset_map[offset] = []
                offset_map[offset].append(note)

        for offset, notes in offset_map.items():
            if len(notes) <= 1:
                continue
            notes.sort(key=lambda n: n.pitch.midi)
            fundamental_hz = _midi_to_hz(notes[0].pitch.midi)
            for note in notes[1:]:
                note_hz = _midi_to_hz(note.pitch.midi)
                for harmonic in [2, 3, 4, 5]:
                    expected_hz = fundamental_hz * harmonic
                    if 0.95 <= note_hz / expected_hz <= 1.05:
                        try:
                            measure.remove(note, recurse=True)
                        except Exception:
                            pass
                        break


def _fix_octave(part):
    for note in part.recurse().notes:
        if not hasattr(note, 'pitch'):
            continue
        while note.pitch.midi > 55:
            note.pitch.midi -= 12
        while note.pitch.midi < 28:
            note.pitch.midi += 12


async def audio_to_musicxml(audio_path: str, instrument: str = "bass") -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe, audio_path, instrument)


def _transcribe(audio_path: str, instrument: str) -> str:
    model_output, midi_data, note_events = predict(
        audio_path,
        get_loaded_model(),
        onset_threshold=0.6,
        frame_threshold=0.2,
        minimum_note_length=100,
        minimum_frequency=40,
        maximum_frequency=400,
    )

    midi_path = audio_path.replace(".wav", ".mid")
    midi_data.write(midi_path)

    score = music21.converter.parse(midi_path)
    score.quantize([4, 8], processOffsets=True, inPlace=True)

    if instrument == "bass":
        for part in score.parts:
            _remove_harmonics(part)
            _fix_octave(part)
            for measure in part.getElementsByClass("Measure"):
                voices = measure.getElementsByClass("Voice")
                if len(voices) > 1:
                    best_voice = max(
                        voices,
                        key=lambda v: sum(
                            n.volume.velocity or 0
                            for n in v.recurse().notes
                        )
                    )
                    for voice in list(voices):
                        if voice is not best_voice:
                            measure.remove(voice)
                    for el in list(best_voice.elements):
                        measure.insert(el.offset, el)
                    measure.remove(best_voice)

    if instrument == "bass":
        inst = music21.instrument.ElectricBass()
    else:
        inst = music21.instrument.Piano()

    if score.parts:
        score.parts[0].insert(0, inst)

    output_path = audio_path.replace(".wav", ".xml")
    score.write("musicxml", fp=output_path)

    return output_path