# Spectral Neighbor Plus
This experiment models musical similarity by translating sound into a set of features that reflect how humans actually experience music.

**PROJECT OVERVIEW**

This repository implements an extended version of my spectral neighbor recommender: a content based music recommendation prototype that retrieves similar tracks using hand crafted audio features and cosine similarity. The system starts from a core spectral feature pipeline and then enriches it with additional descriptors that better capture timbre, motion, texture, harmony, and groove. The goal is not to build a full production recommender but to design and document an interpretable feature space that behaves more like a musical ear than a numeric matcher.

**Audio Processing Pipeline**

All tracks are processed as thirty second audio excerpts at a uniform sampling rate of 22 050 Hz. Each clip is converted into a time–frequency representation using a short time Fourier transform with a 2048 point FFT, a hop length of 512 samples, and a Hann analysis window. From this complex spectrogram a power spectrogram is derived and then mapped to a mel scaled spectrogram. The mel spectrogram is converted to log mel coefficients, and these form the basis for mel frequency cepstral coefficients as well as several higher level spectral and rhythmic descriptors.
For each feature type, statistics are aggregated across time so that every track is represented by a fixed length feature vector. Similarity between tracks is then computed using cosine similarity in this feature space, and a simple k nearest neighbor search is used to retrieve the most similar items.

**Feature Design**

The feature design is organized around families that reflect musically meaningful aspects of the sound: timbre, timbral motion, spectral shape and texture, rhythmic profile, and (optionally) harmonic space. Below, each family starts from the features already present in the original prototype and then introduces the extensions used in this repository.
1. Timbral Core: MFCCs (Existing Prototype)

In the original prototype, timbre is represented by thirteen mel frequency cepstral coefficients computed on log mel spectrogram frames. These coefficients provide a compact summary of the spectral envelope and capture whether a sound feels bright or dark, warm or thin, smooth or harsh. They are widely used in speech and music processing because they approximate how the human ear perceives spectral shape.
For each track, the prototype aggregates the MFCCs across time using the mean and standard deviation of each coefficient. This yields a low dimensional vector that describes the average timbral color and its overall variability throughout the clip.

2. Timbral Motion: Delta and Delta–Delta MFCCs (New in This Repository)

Building on the static MFCCs, this repository introduces first and second order temporal derivatives of the MFCCs, often called delta and delta–delta coefficients. The first order deltas measure how each MFCC changes from one frame to the next, while the second order deltas measure how that change itself accelerates or decelerates over time.
Where the original MFCC block captures “what the sound is” on average, the delta and delta–delta MFCCs capture “how the sound moves.” They distinguish sustained, legato textures from chopped, transient rich material, and they highlight expressive inflections such as evolving pads, vocal slides, and drum fills. After computing these derivatives over frames, the system again aggregates them with mean and standard deviation, so that for every track there is a static MFCC summary, a velocity summary, and an acceleration summary of timbre.
Algorithmically, the addition of these derivative features does not change the recommender’s structure, but it reshapes the geometry of the feature space. Nearest neighbors become songs whose timbral color and the way that color evolves over time are both similar, which generally leads to more coherent and intuitive recommendations.

3. Spectral Shape: Centroid, Rolloff, Flatness, Zero Crossing Rate (Existing Prototype)

The prototype already includes a set of classical spectral descriptors that summarize the overall shape of the spectrum.
**Spectral centroid** tracks the “center of mass” of the spectrum and is often correlated with perceived brightness.
**Spectral rolloff** measures the frequency below which a fixed percentage of total spectral energy lies, which also carries information about brightness and high frequency content.
**Spectral flatness** indicates whether a frame behaves more like a tone (peaky, harmonic) or noise (flat spectrum).
Z**ero crossing rate** gives a simple measure of how often the waveform changes sign, which correlates with noisiness and high frequency activity.
These features are also aggregated with mean and standard deviation, and together they provide a coarse but useful description of whether a track is dark or bright, tonal or noisy, and relatively smooth or texturally busy.

4. Spectral Texture and Weight: Contrast and Bandwidth (New Enhancements)

To deepen the description of spectral shape, the extended system adds spectral contrast and spectral bandwidth. Spectral contrast measures the difference between spectral peaks and valleys in different frequency bands. It captures the sense of whether a mix is dense and full, with strong resonant peaks, or hollow and smooth, with less pronounced structure across the spectrum. Spectral bandwidth describes how widely energy is spread around the centroid and thus helps distinguish narrow, focused sounds from broad, wideband ones.
These descriptors complement the existing centroid and rolloff features by emphasizing texture and body rather than only brightness. Two tracks may share similar centroids and rolloff values yet feel very different: one may be a sparse, intimate vocal with gentle accompaniment, while the other is a thick, saturated mix with strong drums and layered instruments. The contrast and bandwidth features help the recommender recognize and respect these differences. As before, mean and standard deviation provide a compact summary of their behavior across the clip.

5. Rhythmic Profile: Global Tempo (Existing Prototype)

The original prototype estimates a single global tempo per track using onset strength and beat tracking. This scalar gives the recommender a basic sense of how fast or slow a track is in beats per minute and helps separate broad tempo regions such as downtempo R and B, mid tempo hip hop, and faster dance or Afrobeats material.
Although global tempo alone is useful, it is still a very coarse description of rhythmic behavior, since many songs share tempo but feel very different in density and groove.

6. Groove and Beat Level Dynamics: Tempo Stability and Onset Density (New Enhancements)

To enrich the rhythmic description without making the system overly complex, the extended feature set adds a small block of beat level rhythm features. First, tempo estimates over time are examined to derive a measure of tempo stability, which indicates whether the perceived tempo is steady or fluctuating across the clip. Second, onset density is computed as the average number of significant onsets per second or per beat, reflecting how rhythmically dense or sparse the arrangement is. Finally, the average beat strength is captured to summarize how strongly the main pulse is articulated.
In parallel, a beat synchronous pooling strategy is used for key feature families such as MFCCs and chroma (when enabled). Instead of aggregating features over arbitrary fixed length STFT frames, features are averaged over detected beats and then summarized with statistics across beats. This makes the representation less sensitive to fine grained frame alignment and more aligned with how humans perceive rhythm in terms of beats and bars.
Together, these additions allow the recommender to differentiate tracks that share nominal tempo but have very different rhythmic feel. A slow, spacious ballad and a slow but rhythmically dense groove can now fall into different regions of the feature space, leading to neighbor sets that are better matched in their bodily and dance related feel.

7. Harmonic Space: Chroma and Tonal Centroids
   
The current prototype primarily focuses on timbral and spectral based similarity, but the repository is designed to support an optional harmonic feature block based on chroma and tonal centroids (Tonnetz). Chroma features describe the distribution of energy across the twelve pitch classes, independent of octave, and provide a compact representation of key, chordal content, and harmonic emphasis. Tonal centroid features map these chroma vectors into a continuous space where distances correspond more directly to musically meaningful relationships between keys and chords.
When enabled, these features allow the model to take the harmonic and emotional “center” of a track into account. Tracks that share similar keys or move through related chord progressions tend to cluster more closely, while tracks in distant tonal regions drift apart even if their timbres align. This extension is especially useful for applications where harmonic coherence between recommendations and seeds is important, such as DJ style transitions or playlist curation based on mood and key.

8. Lyrical Semantics: Meaning, Mood, and Narrative Content

To complement the sound centered feature space, the extended prototype incorporates a brief block of lyrical semantic features. These features are not based on audio but on the textual content of lyrics and aim to capture what a song communicates rather than how it sounds. The representation draws on approaches from the literature on lyrics based music classification and includes embedded representations of sentiment, mood oriented vocabulary, and thematic content. Sentence level embeddings are computed to obtain a global semantic vector that describes the emotional and narrative profile of the lyrics. Songs with similar lyrical themes, emotional tone, or narrative patterns therefore cluster more closely, even when their timbres or rhythmic profiles differ. This block is intentionally lightweight in the README because a full literature review will be documented in a dedicated file. In the system’s overall geometry, lyrical semantics introduces an additional axis of similarity that connects tracks by meaning, imagery, and affective content, allowing for more human aligned and context aware recommendations.

**Impact on Nearest Neighbor Retrieval**

Across all of these feature families, the underlying algorithm remains deliberately simple. Each track is encoded as a single fixed length feature vector, features are standardized, and cosine similarity is used to measure distances in this space. The k nearest neighbors of a seed track are returned as recommendations.
What changes as features are added is not the algorithm but the shape of the space in which it operates. Static MFCCs and basic spectral descriptors produce neighborhoods dominated by overall timbral color and brightness. Adding delta and delta–delta MFCCs makes neighbors sensitive to the motion and expressiveness of that timbre. Adding spectral contrast and bandwidth introduces a notion of thickness and texture. Adding richer rhythm features encourages tracks with similar groove and density to cluster. Optional harmonic features can then further organize this space by emotional key and tonal relationships.
In combination, these choices aim to produce neighbor sets that feel musically coherent: songs that sound similar, move similarly, and live in related rhythmic and, when enabled, harmonic worlds. The repository is intended as a transparent, well documented playground for exploring how each feature family reshapes this geometry and what “similarity” really means in a musical context.


## Lyrical Semantic Features (TF–IDF)

In addition to spectral audio features, Spectral Neighbor Plus includes a first layer of lyrical semantics. Each track is represented by a TF–IDF vector derived from its full lyrics. This allows the system to consider not only how a song sounds, but also what it is about.

### Data

Raw lyrics are stored in `data/raw/lyrics_raw.csv` with schema:

- `track_id`: identifier consistent with `features_audio.csv`.
- `lyrics`: full lyric text.
- `source` (optional): origin of the lyrics (e.g., Genius, AZLyrics).

Not all tracks are required to have lyrics; tracks without lyrics fall back to audio-only similarity.

### Feature Extraction

Lyrical features are built using a `TfidfVectorizer`:

- Text is lowercased, punctuation is removed, and simple tags such as `[chorus]` and `[verse]` are stripped.
- English stopwords are removed.
- The vocabulary is pruned using:
  - `min_df` to drop extremely rare terms,
  - `max_df` to drop extremely frequent terms,
  - `max_features` to cap dimensionality.
- Both unigrams and bigrams are included (`ngram_range=(1, 2)`).

The resulting TF–IDF matrix is L2-normalized and saved to:

- `data/processed/lyrics_tfidf.npz` – sparse TF–IDF matrix (tracks × terms).
- `data/processed/lyrics_tfidf_vectorizer.joblib` – fitted vectorizer (vocabulary + IDF).
- `data/processed/lyrics_track_ids.csv` – row index to `track_id` mapping.

The pipeline is implemented in `lyrics_features.py`. To build the lyrical features:

```bash
python -m src.lyrics_features
