import { GroupedSelectOption, SelectOption } from "@/types";

type CaptionGroup = 'image' | 'music' | 'video' | 'image/video/sound';
type AdditionalSections = 'caption.model_name_or_path2' | 'caption.caption_prompt' | 'caption.max_res' | 'caption.max_new_tokens' | 'caption.fixed_caption' | 'caption.thinking' | 'caption.batch_size' | 'caption.layer_offloading';

export interface CaptionOption {
    name: string;
    label: string;
    group: CaptionGroup;
    hasMultiLinePrompts?: boolean;
    minNewTokens?: number;
    defaults?: { [key: string]: any };
    additionalSections?: AdditionalSections[];
    name_or_path_options?: SelectOption[];
    name_or_path2_options?: SelectOption[];
    // named caption prompts the user can swap between; the picker only shows
    // when there is more than one, and selecting one just fills caption_prompt
    captionPrompts?: { [name: string]: string };
}

const defaultNameOrPath = '';

const extensionsAudio = ['mp3', 'wav', 'flac', 'ogg'];
const extensionsImage = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];
const extensionsVideo = ['mp4', 'mov', 'webm', 'mkv', 'avi'];

const defaultExtensions = [...extensionsImage];

const defaultImageCaptionPrompt = "Caption this image as if you were going to try to generate it with an image generator. Be thurough and describe everything in the image. Be decisive by stating things as they are. Do not say things like \"It appears that\" Or \"possibly\". Start out with things like \"A person on the beach\" or \"A black dragon\". No preamble. Just get to the point.";

const defaultVideoCaptionPrompt = "Caption this video as if you were going to try to generate it with a video generator. Describe the visual content, how it moves and changes over time, and the camera work. Also describe the audio, including any speech, music, or sound effects, and transcribe spoken dialogue verbatim in quotes. Be decisive by stating things as they are. Do not say things like \"It appears that\" Or \"possibly\". No preamble. Just get to the point.";

// Captions videos as MiniMax T2VA training prompts, following the official
// video prompt writing guide (MiniMaxAI/MiniMax-H3 docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md):
// three fields, [Shot N] timeline with cut timestamps, controlled camera-motion
// vocabulary, (S1) speaker IDs with verbatim <d>[Language] ...</d> dialogue,
// quoted on-screen text, and separate soundscape / non-diegetic music fields.
const minimaxT2VCaptionPrompt = `Caption this video as a MiniMax text-to-video-audio (T2VA) training prompt. Watch the video and listen to the audio, then output exactly three fields in this order, each starting on its own line with these exact field names:

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

Rules for integrated_multimodal_description: Start with [Shot 1] (the first shot gets no timestamp) and state the overall visual style (Live-action, cinematic, 2D-animated, 3D CG, claymation, watercolor, or vintage film) and the initial framing. Then describe everything visible and audible in chronological order: subject appearance and position, scene and key props, actions and reactions, shot changes, speech, and the diegetic sounds that accompany them. Begin each later shot as "[Shot 2] At 00:03.500, the camera cuts to ..." with strictly increasing cut times within the video. Write camera motion as a natural action using this vocabulary: zoom in/out, push in, pull out, pan left/right, truck left/right, tilt up/down, pedestal up/down, arc shot, tracking shot, static shot, POV, roll clockwise/counterclockwise, shake slightly/strongly - adding "with small amplitude"/"with large amplitude" and "at slow speed"/"at fast speed" only when the range or pacing is notable. Give every person who speaks or sings a stable ID like (S1) or (S2); when a speaker first appears, identify them with type, age, gender, and voice quality (pitch, timbre, pace, accent). Transcribe all speech and lyrics verbatim in this form: The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d> - the language tag names the spoken language and only the exact spoken words go inside <d></d>. For narration with no visible lip movement, write "says in an off-screen voiceover:" before the <d> block and state afterwards that the character's lips remain completely closed. Put any legible on-screen text (signs, labels, subtitles) in double quotation marks verbatim without translating it.

overall_soundscape: 1-4 sentences in one paragraph summarizing the ambient sound, physical action sounds, and non-verbal human sounds across the whole video (wind, traffic, footsteps, fabric, impacts, breathing, laughter). Do not repeat dialogue, singing, or diegetic music here. Use N/A only if the video is completely silent.

non_diegetic_music: 1-3 sentences describing background music the characters cannot hear: instrumentation, tempo, rhythm, and dynamic changes, with no abstract mood words. Music coming from a source inside the scene belongs in the multimodal description instead. Use N/A if there is none.

Describe only what is actually seen and heard. Be decisive. No preamble and no extra text - output only the three fields.`;

// The T2VA format applied to a still image: one static [Shot 1], no timeline,
// no audio fields beyond the required N/A placeholders.
const minimaxImageCaptionPrompt = `Caption this image as a MiniMax training prompt for a single still frame. Output exactly three fields in this order, each starting on its own line with these exact field names:

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: N/A

non_diegetic_music: N/A

Rules for integrated_multimodal_description: Write a single [Shot 1] with no timestamp, no cuts, and no camera motion - the camera holds a static shot on a still frame. State the overall visual style first (Live-action, cinematic, 2D-animated, 3D CG, claymation, watercolor, or vintage film) and the framing (extreme wide shot, wide shot, medium shot, medium close-up, close-up, or extreme close-up, plus the viewpoint). Then describe everything visible decisively and specifically: subject appearance, pose, and position; clothing, colors, and materials; the scene, lighting, and key props; and the spatial relationships between them. There is no motion, no dialogue, and no sound - do not invent any. Put any legible text in the image (signs, labels, packaging) in double quotation marks verbatim without translating it.

overall_soundscape and non_diegetic_music are always exactly N/A for a still image.

Describe only what is actually visible. Be decisive. No preamble and no extra text - output only the three fields.`;

// Editable ADDITIONAL INSTRUCTIONS block injected into the Ideogram system prompt.
// Users can tweak this for dataset-specific guidance without altering the fixed
// output contract, element/background rules, or bbox format.
const defaultIdeogramCaptionPrompt = "Describe only what is actually visible in the image — never invent, add, or infer content that is not present. Identify the medium (photograph, illustration, 3D render, or graphic design) and name any recognizable people, brands, characters, or landmarks. Be decisive and specific: commit to one concrete value per attribute (one color, one material, one pose) and avoid hedging. Transcribe every piece of legible text verbatim.";

export const captionerTypes: CaptionOption[] = [
    {
        name: 'AceStepCaptioner',
        label: 'Ace Step',
        group: 'music',
        defaults: {
            'config.process[0].caption.model_name_or_path': ['ACE-Step/acestep-transcriber', defaultNameOrPath],
            'config.process[0].caption.model_name_or_path2': ['ACE-Step/acestep-captioner', undefined],
            'config.process[0].caption.extensions': [extensionsAudio, defaultExtensions],
        },
        name_or_path_options: [
            { value: 'ACE-Step/acestep-transcriber', label: 'ACE-Step/acestep-transcriber' },
        ],
        name_or_path2_options: [
            { value: 'ACE-Step/acestep-captioner', label: 'ACE-Step/acestep-captioner' },
        ],
        additionalSections: [
            'caption.model_name_or_path2',
            'caption.fixed_caption',
        ],
    },
    {
        name: 'Qwen3VLCaptioner',
        label: 'Qwen3-VL',
        group: 'image',
        defaults: {
            'config.process[0].caption.model_name_or_path': ['Qwen/Qwen3-VL-8B-Instruct', defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultImageCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [512, undefined],
            'config.process[0].caption.max_new_tokens': [128, undefined],

        },
        name_or_path_options: [
            { value: 'Qwen/Qwen3-VL-2B-Instruct', label: 'Qwen/Qwen3-VL-2B-Instruct' },
            { value: 'Qwen/Qwen3-VL-4B-Instruct', label: 'Qwen/Qwen3-VL-4B-Instruct' },
            { value: 'Qwen/Qwen3-VL-8B-Instruct', label: 'Qwen/Qwen3-VL-8B-Instruct' },
            { value: 'huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated', label: 'huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated' },
            { value: 'Qwen/Qwen3-VL-30B-A3B-Instruct', label: 'Qwen/Qwen3-VL-30B-A3B-Instruct' },
            { value: 'Qwen/Qwen3.6-27B', label: 'Qwen/Qwen3.6-27B' },
            { value: 'huihui-ai/Huihui-Qwen3.6-27B-abliterated', label: 'huihui-ai/Huihui-Qwen3.6-27B-abliterated' },
        ],
        additionalSections: [
            'caption.caption_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
            'caption.thinking',
        ],
    },
    {
        name: 'Qwen3OmniCaptioner',
        label: 'Qwen3-Omni',
        group: 'image/video/sound',
        defaults: {
            'config.process[0].caption.model_name_or_path': ['ai-toolkit/Qwen3-Omni-30B-A3B-Thinking', defaultNameOrPath],
            'config.process[0].caption.extensions': [[...extensionsVideo, ...extensionsImage], defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultVideoCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [512, undefined],
            'config.process[0].caption.max_new_tokens': [512, undefined],
            'config.process[0].caption.batch_size': [1, undefined],
            'config.process[0].caption.compile': [true, false],
        },
        name_or_path_options: [
            { value: 'ai-toolkit/Qwen3-Omni-30B-A3B-Instruct', label: 'ai-toolkit/Qwen3-Omni-30B-A3B-Instruct' },
            { value: 'ai-toolkit/Qwen3-Omni-30B-A3B-Thinking', label: 'ai-toolkit/Qwen3-Omni-30B-A3B-Thinking' },
            { value: 'ai-toolkit/Huihui-Qwen3-Omni-30B-A3B-Thinking-abliterated', label: 'ai-toolkit/Huihui-Qwen3-Omni-30B-A3B-Thinking-abliterated' },
        ],
        captionPrompts: {
            'General': defaultVideoCaptionPrompt,
            'MiniMax H4 T2V': minimaxT2VCaptionPrompt,
            'MiniMax H4 Image': minimaxImageCaptionPrompt,
        },
        additionalSections: [
            'caption.caption_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
            'caption.batch_size',
            'caption.layer_offloading',
            'caption.thinking',
        ],
    },
    {
        name: 'Ideogram4Captioner',
        label: 'Ideogram 4 Captioner',
        group: 'image',
        hasMultiLinePrompts: true,
        // The deconstruction JSON is long; the Python captioner also enforces this floor.
        minNewTokens: 3072,
        defaults: {
            'config.process[0].caption.model_name_or_path': ['Qwen/Qwen3-VL-8B-Instruct', defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultIdeogramCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [512, undefined],
            'config.process[0].caption.max_new_tokens': [4096, undefined],
        },
        name_or_path_options: [
            { value: 'Qwen/Qwen3-VL-2B-Instruct', label: 'Qwen/Qwen3-VL-2B-Instruct' },
            { value: 'Qwen/Qwen3-VL-4B-Instruct', label: 'Qwen/Qwen3-VL-4B-Instruct' },
            { value: 'Qwen/Qwen3-VL-8B-Instruct', label: 'Qwen/Qwen3-VL-8B-Instruct' },
            { value: 'Qwen/Qwen3-VL-30B-A3B-Instruct', label: 'Qwen/Qwen3-VL-30B-A3B-Instruct' },
        ],
        additionalSections: [
            'caption.caption_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
        ],
    },

].sort((a, b) => {
    // Sort by label, case-insensitive
    return a.label.localeCompare(b.label, undefined, { sensitivity: 'base' });
}) as any;

export const groupedCaptionerTypes: GroupedSelectOption[] = captionerTypes.reduce((acc, arch) => {
    const group = acc.find(g => g.label === arch.group);
    if (group) {
        group.options.push({ value: arch.name, label: arch.label });
    } else {
        acc.push({
            label: arch.group,
            options: [{ value: arch.name, label: arch.label }],
        });
    }
    return acc;
}, [] as GroupedSelectOption[]);

export const quantizationOptions: SelectOption[] = [
    { value: '', label: '- NONE -' },
    { value: 'float8', label: 'float8' },
    { value: 'convrot8', label: '8bit convrot (default)' },
    { value: 'convrot4', label: '4bit convrot (nvfp4)' },
    { value: 'convrotint7', label: '7bit convrot' },
    { value: 'convrotint6', label: '6bit convrot' },
    { value: 'convrotint5', label: '5bit convrot' },
    { value: 'convrotint4', label: '4bit convrot' },
    { value: 'convrotint3', label: '3bit convrot' },
    { value: 'convrotint2', label: '2bit convrot' },
    { value: 'convrotbitnet', label: '1.58bit convrot (bitnet)' },
    { value: 'uint7', label: '7 bit' },
    { value: 'uint6', label: '6 bit' },
    { value: 'uint5', label: '5 bit' },
    { value: 'uint4', label: '4 bit' },
    { value: 'uint3', label: '3 bit' },
    { value: 'uint2', label: '2 bit' },
];

export const batchSizeOptions: SelectOption[] = [
    { value: '1', label: '1 (default)' },
    { value: '2', label: '2' },
    { value: '4', label: '4' },
    { value: '8', label: '8' },
    { value: '12', label: '12' },
    { value: '16', label: '16' },
    { value: '24', label: '24' },
    { value: '32', label: '32' },
];

export const maxResOptions: SelectOption[] = [
    { value: '256', label: '256' },
    { value: '512', label: '512 (default)' },
    { value: '768', label: '768' },
    { value: '1024', label: '1024' },
];
export const maxNewTokensOptions: SelectOption[] = [
    { value: '64', label: '64' },
    { value: '128', label: '128 (default)' },
    { value: '256', label: '256' },
    { value: '512', label: '512' },
    { value: '1024', label: '1024' },
    { value: '2048', label: '2048' },
    { value: '3072', label: '3072' },
    { value: '4096', label: '4096' },
    { value: '8192', label: '8192' },
];

export const defaultQtype = 'convrot8';