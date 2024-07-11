from magenta.models.music_vae.trained_model import TrainedModel
from magenta.models.music_vae import configs
import magenta.music as mm

def download(note_sequence, filename):
    mm.sequence_proto_to_midi_file(note_sequence, filename)


model = TrainedModel(config=configs.CONFIG_MAP['cat-drums_2bar_small'], batch_size=4, checkpoint_dir_or_path="/workspaces/work/models/cat-drums_2bar_small.lokl/cat-drums_2bar_small.lokl.ckpt.index")
samples = model.sample(n=4, length=32, temperature=0.5)
for i, ns in enumerate(samples):
    download(ns, '%s_sample_%d.mid' % (model, i))
