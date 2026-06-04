import { SampleConfig } from "@/types";

export const defaultSampleConfig: SampleConfig = {
  sampler: 'flowmatch',
  sample_every: 250,
  width: 1024,
  height: 1024,
  samples: [
    {
      prompt: 'woman with red hair, playing chess at the park, bomb going off in the background',
    },
    {
      prompt: 'a woman holding a coffee cup, in a beanie, sitting at a cafe',
    },
    {
      prompt: 'a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini',
    },
    {
      prompt:
        'a man showing off his cool new t shirt at the beach, a shark is jumping out of the water in the background',
    },
    {
      prompt: 'a bear building a log cabin in the snow covered mountains',
    },
    {
      prompt: 'woman playing the guitar, on stage, singing a song, laser lights, punk rocker',
    },
    {
      prompt: 'hipster man with a beard, building a chair, in a wood shop',
    },
    {
      prompt:
        'photo of a man, white background, medium shot, modeling clothing, studio lighting, white backdrop',
    },
    {
      prompt: "a man holding a sign that says, 'this is a sign'",
    },
    {
      prompt:
        'a bulldog, in a post apocalyptic world, with a shotgun, in a leather jacket, in a desert, with a motorcycle',
    },
  ],
  neg: '',
  seed: 42,
  walk_seed: true,
  guidance_scale: 4,
  sample_steps: 30,
  num_frames: 1,
  fps: 1,
}

export const defaultAudioSampleConfig: SampleConfig = {
  sampler: 'flowmatch',
  sample_every: 250,
  width: 1024,
  height: 1024,
  samples: [
    {
      prompt: `
<CAPTION>my style song</CAPTION>
<LYRICS>
[Intro choir]
Laura
Laura
Laura
Laura Training

[Verse 1]
A new open model, she been training it nightly
AI tool kit, she configures it tightly,
Loss curves dropping down to the floor
Wondering if she's done or she should train it some more.

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Verse 4]
She's caching all the latents
now she doesn't need a vay
Training on some voices
What will she make them say

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Guitar Solo]

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Outro]
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
</LYRICS>
<BPM>112</BPM>
<KEYSCALE>A minor</KEYSCALE>
<TIMESIGNATURE>4</TIMESIGNATURE>
<DURATION>180</DURATION>
<LANGUAGE>en</LANGUAGE>
`,
    }, {
      prompt: `
<CAPTION>my style song</CAPTION>
<LYRICS>
[Intro choir]
Laura
Laura
Laura
Laura Training

[Verse 1]
A new open model, she been training it nightly
AI tool kit, she configures it tightly,
Loss curves dropping down to the floor
Wondering if she's done or she should train it some more.

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Verse 4]
She's caching all the latents
now she doesn't need a vay
Training on some voices
What will she make them say

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Guitar Solo]

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Outro]
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
</LYRICS>
<BPM>112</BPM>
<KEYSCALE>A minor</KEYSCALE>
<TIMESIGNATURE>4</TIMESIGNATURE>
<DURATION>180</DURATION>
<LANGUAGE>en</LANGUAGE>
`,
    }, {
      prompt: `
<CAPTION>my style song</CAPTION>
<LYRICS>
[Intro choir]
Laura
Laura
Laura
Laura Training

[Verse 1]
A new open model, she been training it nightly
AI tool kit, she configures it tightly,
Loss curves dropping down to the floor
Wondering if she's done or she should train it some more.

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Verse 4]
She's caching all the latents
now she doesn't need a vay
Training on some voices
What will she make them say

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Guitar Solo]

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Outro]
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
</LYRICS>
<BPM>112</BPM>
<KEYSCALE>A minor</KEYSCALE>
<TIMESIGNATURE>4</TIMESIGNATURE>
<DURATION>180</DURATION>
<LANGUAGE>en</LANGUAGE>
`,
    }, {
      prompt: `
<CAPTION>my style song</CAPTION>
<LYRICS>
[Intro choir]
Laura
Laura
Laura
Laura Training

[Verse 1]
A new open model, she been training it nightly
AI tool kit, she configures it tightly,
Loss curves dropping down to the floor
Wondering if she's done or she should train it some more.

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Verse 4]
She's caching all the latents
now she doesn't need a vay
Training on some voices
What will she make them say

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Guitar Solo]

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Chorus]
Laura training
She trains on what she pleases
Laura training
No paying corporate sleazes
Laura training
This could be her best one
Why go outside, Laura training is too fun

[Instrumental Break]

[Outro]
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
Ah yeah!
It's Converging!
</LYRICS>
<BPM>112</BPM>
<KEYSCALE>A minor</KEYSCALE>
<TIMESIGNATURE>4</TIMESIGNATURE>
<DURATION>180</DURATION>
<LANGUAGE>en</LANGUAGE>
`,
    },
  ],
  neg: '',
  seed: 42,
  walk_seed: true,
  guidance_scale: 4,
  sample_steps: 30,
  num_frames: 1,
  fps: 1,
};


export const defaultIdeogramSamplesConfig: SampleConfig = {
  sampler: 'flowmatch',
  sample_every: 250,
  width: 1024,
  height: 1024,
  samples: [
    {
      prompt: `
{
  "high_level_description": "A red-haired woman calmly playing chess at a wooden table in a sunlit public park during late afternoon, while a violent orange explosion erupts among distant trees behind her, creating a surreal contrast between serene focus and sudden chaos.",
  "style_description": {
    "aesthetics": "moody cinematic realism, sharp foreground focus with hazy atmospheric background, dramatic contrast",
    "lighting": "warm low-angle afternoon sun from the right, intense orange glow from the distant blast adding a secondary warm rim light, long soft shadows",
    "photo": "full-frame DSLR, 50mm lens, f/2.0, shallow depth of field with motion blur in the background blast",
    "medium": "photography",
    "color_palette": ["#1F3D1A", "#C9521E", "#E8B14C", "#9FB37A", "#3A2A1F", "#D9C9A8"]
  },
  "compositional_deconstruction": {
    "background": "An open green park in late afternoon, mown grass receding into the distance, a gravel path curving away to the right, scattered mature trees along the horizon under a hazy pale-blue sky. Atmospheric haze and drifting smoke soften the far distance, with sunlight filtering warmly across the open space to create depth between the near table and the far tree line.",
    "elements": [
      {
        "type": "obj",
        "bbox": [560, 120, 880, 430],
        "desc": "A large explosion erupting among the distant trees in the upper-right background, a billowing fireball of orange and yellow flame with a rising column of dark grey-black smoke and debris flung outward. Soft-focused and motion-blurred to read as far away, casting a warm glow that bleeds into the surrounding haze.",
        "color_palette": ["#C9521E", "#E8B14C", "#3A2A1F", "#7A6A55"]
      },
      {
        "type": "obj",
        "bbox": [40, 300, 360, 470],
        "desc": "A cluster of mid-distance park trees along the left horizon, full green canopies catching warm afternoon light on their right edges, partially obscured by drifting smoke from the blast. Slightly soft-focused to sit in the midground behind the table.",
        "color_palette": ["#1F3D1A", "#9FB37A", "#5C7A3A"]
      },
      {
        "type": "obj",
        "bbox": [180, 560, 820, 1000],
        "desc": "A weathered wooden park table occupying the lower portion of the frame, viewed slightly from the side, grain and knots visible on the planks, warm sunlight raking across the surface from the right and casting a long shadow to the left. Sharp focus as the anchoring foreground surface.",
        "color_palette": ["#5A3A22", "#8A6038", "#D9C9A8"]
      },
      {
        "type": "obj",
        "bbox": [340, 600, 680, 760],
        "desc": "A checkered chessboard resting flat on the table in the lower-center of the frame, mid-game with carved wooden pieces casting small crisp shadows in the afternoon light, a few captured pieces set off to one side. Tilted slightly toward the viewer, in sharp focus.",
        "color_palette": ["#E8E0CC", "#3A2A1F", "#8A6038"]
      },
      {
        "type": "obj",
        "bbox": [330, 250, 640, 760],
        "desc": "A woman with vivid red hair seated at the far side of the table, center-left, leaning forward with one hand resting near a chess piece, her gaze fixed calmly downward on the board, fully absorbed and unaware of the chaos behind her. Wavy shoulder-length hair catching warm light and a faint glow from the distant fire, wearing a simple olive jacket, rendered in sharp focus as the primary subject.",
        "color_palette": ["#B0301F", "#C9521E", "#6E7A4A", "#E2C2A0"]
      },
      {
        "type": "obj",
        "bbox": [680, 540, 880, 900],
        "desc": "An empty weathered wooden chair on the near right side of the table, angled slightly outward as if its occupant just left, sunlight warming its right edge and a long shadow falling across the grass beside it. Slightly softer focus than the woman, in the right foreground.",
        "color_palette": ["#5A3A22", "#8A6038", "#3A2A1F"]
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "high_level_description": "A young woman in a knit beanie sits at a cozy café table on an overcast morning, cradling a warm coffee cup in both hands and gazing thoughtfully toward the window light, evoking a calm, intimate, contemplative mood.",
  "style_description": {
    "aesthetics": "moody cinematic lifestyle photography, sharp focus on subject, shallow depth of field with creamy bokeh",
    "lighting": "soft diffused natural window light from the left, gentle cool daylight key with warm ambient fill from interior tungsten",
    "photo": "50mm prime lens, f/1.8, slight film grain, DSLR full-frame",
    "medium": "photography",
    "color_palette": ["#3E2C20", "#A9876B", "#D8C4A8", "#6E5A48", "#C24A2E", "#E9E2D6"]
  },
  "compositional_deconstruction": {
    "background": "The blurred interior of a rustic café: warm wooden wall paneling and a softly out-of-focus window on the left spilling pale daylight, with hanging pendant lights and shelves of jars receding into a hazy bokeh on the right, creating a sense of warm enclosed depth.",
    "elements": [
      {
        "type": "obj",
        "bbox": [40, 110, 360, 640],
        "desc": "A tall frosted café window in the midground left, glowing with diffuse overcast daylight, condensation faintly softening its surface and acting as the primary light source washing across the scene.",
        "color_palette": ["#E9E2D6", "#C9C0B2", "#9AA0A0"]
      },
      {
        "type": "obj",
        "bbox": [560, 230, 940, 600],
        "desc": "A weathered wooden shelf along the back wall on the right, holding a few out-of-focus glass jars and a small potted plant, rendered as soft warm bokeh that anchors the depth behind the subject.",
        "color_palette": ["#6E5A48", "#8B7152", "#4A3A2C"]
      },
      {
        "type": "obj",
        "bbox": [240, 250, 760, 1000],
        "desc": "A young woman seated at the table, occupying the central-lower frame and turned slightly toward the window at left, wearing a chunky cream-and-rust knit beanie pushed back over wavy hair, an oversized oatmeal sweater, her shoulders relaxed and her gaze directed softly off-frame left, catching the cool window light along her cheek and jaw.",
        "color_palette": ["#A9876B", "#D8C4A8", "#C24A2E", "#3E2C20"]
      },
      {
        "type": "obj",
        "bbox": [400, 600, 640, 850],
        "desc": "A rounded ceramic coffee cup held in both of the woman's hands near her chest, foreground center, thin curl of steam rising from the dark surface, the warm interior light catching the glazed rim and her fingers wrapped snugly around it.",
        "color_palette": ["#3E2C20", "#C9BBA6", "#7A5A40"]
      },
      {
        "type": "obj",
        "bbox": [180, 780, 880, 1000],
        "desc": "A rustic dark-wood café table filling the foreground edge, its grain catching a soft sheen, with a faint coffee ring and a folded napkin resting near the bottom corner, grounding the composition.",
        "color_palette": ["#4A3526", "#6E5238", "#2C1F16"]
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "high_level_description": "A confident horse works the decks as a DJ in a packed underground night club at peak hour, surrounded by laser beams and rolling fog, holding a martini glass in one hoof while the crowd glows in the haze.",
  "style_description": {
    "aesthetics": "high-energy nightlife photography, surreal and playful, deep shadows with neon punch, slight chromatic aberration",
    "lighting": "low ambient darkness cut by saturated laser beams and colored wash from above, strobing key light on the subject, cool magenta and cyan rim light with warm amber spill",
    "photo": "fisheye lens, ~8mm, exaggerated barrel distortion, f/2.8, slight motion blur in the haze, faint sensor grain",
    "medium": "photography",
    "color_palette": ["#0A0410", "#E0146E", "#16D6E8", "#7A1FB0", "#F2B705", "#1B0A2E"]
  },
  "compositional_deconstruction": {
    "background": "A dark, cavernous club interior with a low ceiling rig of moving heads and laser projectors, distant blurred crowd silhouettes packed toward a back bar, dense low-lying fog drifting across the floor, the whole space curved and stretched by the fisheye so vertical lines bow outward toward the frame edges.",
    "elements": [
      {
        "type": "obj",
        "bbox": [0, 30, 1000, 520],
        "desc": "Crisscrossing green and magenta laser beams fanning out from ceiling projectors at the top of the frame, splaying across the upper third and bending along the fisheye curve, catching in the fog to form bright volumetric sheets and dotted scatter planes.",
        "color_palette": ["#16D6E8", "#E0146E", "#3DF27A", "#7A1FB0"]
      },
      {
        "type": "obj",
        "bbox": [120, 380, 880, 820],
        "desc": "Thick rolling smoke-machine fog blanketing the midground and floor, lit through by colored beams, denser at center where it glows and thinning toward the warped edges, partially veiling the lower body of the subject.",
        "color_palette": ["#2A1140", "#7A1FB0", "#16D6E8", "#1B0A2E"]
      },
      {
        "type": "obj",
        "bbox": [310, 250, 760, 640],
        "desc": "A glossy black DJ booth and mixer console set across the lower-center foreground, twin turntables and a lit channel mixer with glowing faders and knobs, front edge bulging toward the viewer from the fisheye distortion, reflective surface catching pink and cyan highlights.",
        "color_palette": ["#0A0410", "#E0146E", "#16D6E8", "#F2B705"]
      },
      {
        "type": "obj",
        "bbox": [270, 90, 720, 560],
        "desc": "A chestnut-brown horse standing as the DJ behind the booth, head and torso filling the center frame, leaning into one turntable with a front hoof, mane catching colored rim light, ears forward and gaze toward the crowd, slight beam glare across its glossy coat.",
        "color_palette": ["#7A4B2A", "#3A2414", "#E0146E", "#16D6E8"]
      },
      {
        "type": "obj",
        "bbox": [640, 300, 820, 560],
        "desc": "A classic martini glass held aloft in the horse's raised right hoof on the mid-right, pale liquid with an olive on a pick, the glass rim and surface flashing cyan and magenta laser glints.",
        "color_palette": ["#E8E2C0", "#16D6E8", "#E0146E", "#9BAF3A"]
      },
      {
        "type": "obj",
        "bbox": [0, 560, 1000, 1000],
        "desc": "Foreground crowd of raised hands and blurred dancing silhouettes along the bottom edge, backlit into dark shapes rimmed by neon, hands reaching up into the fog and beams, heavily curved by the fisheye at the frame's lower corners.",
        "color_palette": ["#1B0A2E", "#E0146E", "#16D6E8", "#7A1FB0"]
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "high_level_description": "A cheerful man stands on a sunny beach proudly showing off his new graphic t-shirt while a large shark breaches the ocean surface behind him, blending a relaxed summer mood with a playful jolt of drama.",
  "style_description": {
    "aesthetics": "vibrant lifestyle photography, sharp focus on the subject with a softly blurred background, candid summer energy",
    "lighting": "bright natural midday sunlight from the upper right, warm and slightly hard, with strong highlights on skin and water and gentle fill from sand bounce",
    "photo": "DSLR, 50mm lens, f/4, fast shutter to freeze the breaching shark",
    "medium": "photography",
    "color_palette": ["#3FA9D6", "#F2E2B6", "#1C6E8C", "#E8B04B", "#2B3A42"]
  },
  "compositional_deconstruction": {
    "background": "A wide stretch of bright tropical beach under a clear blue sky with scattered wispy clouds near the horizon; turquoise ocean water fills the middle band of the frame, transitioning to pale golden wet sand in the lower foreground, with soft heat haze and shimmering reflections suggesting strong midday sun and open depth toward the horizon.",
    "elements": [
      {
        "type": "obj",
        "bbox": [560, 300, 800, 470],
        "desc": "A large gray shark breaching upward out of the ocean in the midground right, body angled diagonally with its head clearing the surface and tail still in the water, mouth slightly open, spraying a burst of white sea foam and droplets around it; smaller in scale due to its distance, creating a dramatic but slightly comedic focal accent behind the man.",
        "color_palette": ["#6B7C85", "#9FB3BC", "#FFFFFF", "#1C6E8C"]
      },
      {
        "type": "obj",
        "bbox": [230, 250, 560, 940],
        "desc": "A smiling man standing center-left in the foreground, facing the camera, tugging the hem of his t-shirt outward with both hands to show it off, weight on one leg in a relaxed casual stance; tanned skin catching warm sunlight, wind lightly moving his hair, gaze directed at the viewer with an enthusiastic expression, occupying the dominant vertical mass of the frame.",
        "color_palette": ["#D9A66C", "#3B3027", "#2B3A42", "#F2E2B6"]
      },
      {
        "type": "obj",
        "bbox": [290, 470, 510, 700],
        "desc": "The man's new graphic t-shirt, a vivid printed cotton tee stretched outward by his hands to display its bold front design, fabric slightly creased from being pulled, catching crisp highlights along the folds and reading as the proud centerpiece of his pose.",
        "color_palette": ["#E84C3D", "#FFFFFF", "#2B3A42", "#E8B04B"]
      },
      {
        "type": "obj",
        "bbox": [0, 760, 1000, 1000],
        "desc": "A band of damp golden beach sand sweeping across the lower foreground, dotted with faint footprints and tiny shell fragments, surface glinting where the wet sheen reflects the bright sky, anchoring the man and adding tactile depth at the bottom of the frame.",
        "color_palette": ["#E8D6A6", "#C9A86A", "#F5EBD0"]
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "high_level_description": "A large brown bear works on constructing a half-built log cabin in a snow-covered mountain valley on a crisp, overcast winter morning, conveying a cozy yet industrious, slightly whimsical wilderness mood.",
  "style_description": {
    "aesthetics": "warm storybook digital illustration, painterly textures, gentle stylization with soft edges and rich detail",
    "lighting": "soft diffused overcast daylight from the upper left, cool ambient shadows with subtle warm bounce from the wood",
    "photo": "",
    "medium": "digital illustration",
    "color_palette": ["#E8EEF2", "#A9C0D0", "#7A5230", "#3D5A4C", "#C8895A", "#2E3A44"]
  },
  "compositional_deconstruction": {
    "background": "Towering snow-capped mountain peaks recede into a pale grey-blue winter sky, partially veiled by drifting mist. A dense stand of frosted evergreen pines blankets the mid-slope, and an unbroken sheet of fresh snow covers the valley floor, broken only by faint drifts and gentle undulations that establish depth toward the distant ridgeline.",
    "elements": [
      {
        "type": "obj",
        "bbox": [60, 230, 420, 540],
        "desc": "A cluster of tall frosted pine trees standing in the midground left, their dark green boughs heavy with clumps of snow, slightly out of focus to push them back in depth and frame the construction scene.",
        "color_palette": ["#3D5A4C", "#2E3A44", "#E8EEF2"]
      },
      {
        "type": "obj",
        "bbox": [380, 420, 880, 760],
        "desc": "A half-finished log cabin occupying the center-right midground, built of stacked honey-brown timber logs about four courses high with notched corner joints, an open doorway gap on the left side, and a dusting of snow along the top logs; freshly hewn and inviting.",
        "color_palette": ["#7A5230", "#C8895A", "#5A3C22", "#E8EEF2"]
      },
      {
        "type": "obj",
        "bbox": [520, 280, 700, 480],
        "desc": "A large brown bear standing upright on its hind legs in the right midground beside the cabin, gripping a heavy log between its forepaws as it lifts it into place, fur thick and shaggy with snowflakes caught in it, head tilted toward its work with a focused expression.",
        "color_palette": ["#5A3C22", "#3A2818", "#8A6038", "#E8EEF2"]
      },
      {
        "type": "obj",
        "bbox": [150, 700, 470, 850],
        "desc": "A small stack of cut logs and a couple of fresh wood shavings resting on the snow in the foreground left, the sawn ends showing pale concentric rings, a few flecks of sawdust scattered across the surrounding snow.",
        "color_palette": ["#C8895A", "#7A5230", "#EDE3D2"]
      },
      {
        "type": "obj",
        "bbox": [600, 760, 760, 880],
        "desc": "A worn steel axe embedded blade-down in a chopping stump in the lower foreground right, its wooden handle angled up and to the right, light snow gathered on the stump's flat top.",
        "color_palette": ["#6B7178", "#4A3320", "#E8EEF2"]
      },
      {
        "type": "obj",
        "bbox": [0, 0, 1000, 220],
        "desc": "Faint drifting snowflakes and a soft mist layer across the upper portion of the frame, thinning the distant peaks and adding cold atmospheric depth without obscuring the foreground.",
        "color_palette": ["#E8EEF2", "#A9C0D0", "#C4D4DE"]
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "high_level_description": "A punk rock woman performs alone on a dark stage at night, mid-song with her guitar, surrounded by sweeping laser lights and haze that cut through the darkness, charged with raw, electric energy.",
  "style_description": {
    "aesthetics": "moody cinematic concert photography, high contrast, gritty and energetic with motion in the lights",
    "lighting": "dramatic stage lighting from above and behind, vivid colored laser beams cutting through atmospheric haze, hard rim light on the performer, deep shadows",
    "photo": "DSLR concert photograph, 85mm, f/2.0, slight motion blur on the lasers, ISO grain",
    "medium": "photography",
    "color_palette": ["#0B0B12", "#E0204A", "#1FB6C9", "#7A2BD4", "#F2E9D8", "#101830"]
  },
  "compositional_deconstruction": {
    "background": "A deep, near-black stage interior at night, dense with atmospheric haze that catches the light. Vertical truss rigging and dim silhouettes of speaker stacks recede into darkness at the edges. The far depth dissolves into a smoky void that gives the colored beams room to glow.",
    "elements": [
      {
        "type": "obj",
        "bbox": [0, 20, 1000, 620],
        "desc": "A web of vivid laser beams fanning out from rigging high in the frame, sweeping diagonally across the upper two-thirds of the scene. Sharp magenta, cyan, and violet lines pierce the haze, some converging toward the performer, others splaying outward, with slight motion blur conveying their sweep.",
        "color_palette": ["#E0204A", "#1FB6C9", "#7A2BD4", "#F2E9D8"]
      },
      {
        "type": "obj",
        "bbox": [120, 80, 360, 520],
        "desc": "A tall stage spotlight rig at midground left, its lens flaring a hot magenta glow that bleeds into the surrounding haze. Partially silhouetted metal housing, angled down toward center stage.",
        "color_palette": ["#E0204A", "#101830", "#0B0B12"]
      },
      {
        "type": "obj",
        "bbox": [650, 90, 880, 500],
        "desc": "A cluster of cyan-tinted backlights at midground right mounted on dark truss, throwing a cool teal wash and lens flares through the smoke, balancing the warm magenta on the opposite side.",
        "color_palette": ["#1FB6C9", "#101830", "#0B0B12"]
      },
      {
        "type": "obj",
        "bbox": [355, 180, 660, 940],
        "desc": "The punk rocker woman, full-bodied and slightly right of center, captured mid-song. She leans into a microphone on a stand, mouth open singing, eyes intense. Spiked, brightly dyed hair, ripped band tee, studded leather, and torn jeans. Her body is rim-lit by colored stage light against the dark haze, sweat catching highlights, in a dynamic forward-leaning stance.",
        "color_palette": ["#E0204A", "#F2E9D8", "#0B0B12", "#7A2BD4"]
      },
      {
        "type": "obj",
        "bbox": [330, 440, 690, 820],
        "desc": "An electric guitar slung across the woman's torso, held mid-strum with her right hand near the strings and left hand on the fretboard. Glossy body catching sharp magenta and cyan reflections, chrome hardware flaring, positioned diagonally across her lower body.",
        "color_palette": ["#101830", "#1FB6C9", "#E0204A", "#F2E9D8"]
      },
      {
        "type": "obj",
        "bbox": [430, 250, 545, 560],
        "desc": "A chrome microphone on a slim stand directly in front of the singer's face at center frame, catching a bright specular highlight from the stage lights, slightly haloed by the haze.",
        "color_palette": ["#F2E9D8", "#0B0B12", "#1FB6C9"]
      },
      {
        "type": "obj",
        "bbox": [0, 780, 1000, 1000],
        "desc": "The dark, glossy stage floor across the foreground, reflecting smeared streaks of magenta, cyan, and violet from the lasers and the performer above, fading to black at the front edge.",
        "color_palette": ["#0B0B12", "#E0204A", "#1FB6C9", "#7A2BD4"]
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "high_level_description": "A bearded hipster craftsman in a sunlit woodworking shop, focused intently as he assembles a wooden chair surrounded by tools and sawdust, evoking a warm, artisanal, hands-on mood.",
  "style_description": {
    "aesthetics": "warm documentary photography, sharp focus on subject, shallow depth of field, authentic artisanal feel",
    "lighting": "soft natural light streaming from a large window on the left, warm afternoon color temperature with gentle dust particles catching the light",
    "photo": "35mm DSLR, 50mm lens, f/2.0, slight film grain",
    "medium": "photography",
    "color_palette": ["#6B4A2E", "#C9A36B", "#E8D5B0", "#3A2C1F", "#8C7355", "#D9C2A0"]
  },
  "compositional_deconstruction": {
    "background": "A rustic woodworking shop interior with weathered plank walls, a large window on the left admitting bright afternoon light, blurred shelves of tools and lumber stacked in the soft-focus depth, faint sawdust haze drifting through the air, warm earthy tones throughout.",
    "elements": [
      {
        "type": "obj",
        "bbox": [40, 120, 320, 520],
        "desc": "A pegboard and tool rack mounted on the back wall, upper-left of frame, slightly out of focus, holding hand saws, chisels, and clamps hanging in rows, midground depth, muted in the soft background light.",
        "color_palette": ["#5A4630", "#8C7355", "#3A2C1F"]
      },
      {
        "type": "obj",
        "bbox": [620, 250, 980, 720],
        "desc": "A heavy wooden workbench occupying the right side of the frame, its surface scattered with hand tools, wood shavings, and a coffee mug, warm light grazing the worn timber grain, midground to foreground.",
        "color_palette": ["#6B4A2E", "#C9A36B", "#4A3622"]
      },
      {
        "type": "obj",
        "bbox": [300, 160, 640, 760],
        "desc": "A bearded hipster man in his early thirties, center-left of frame, wearing a rolled-sleeve flannel shirt and a leather apron, leaning forward with focused concentration as he works, hair tied back, forearms tensed, side-lit by the window so highlights catch his beard and the dust around him.",
        "color_palette": ["#8C5A3A", "#C9A36B", "#5A3B28", "#E8D5B0"]
      },
      {
        "type": "obj",
        "bbox": [380, 520, 760, 920],
        "desc": "A partially assembled solid-wood chair in the lower-center foreground, frame and three legs joined with a fourth resting nearby, freshly planed pale timber showing clean tool marks, the man's hands fitting a joint, sharply in focus.",
        "color_palette": ["#D9C2A0", "#C9A36B", "#9C7A4E"]
      },
      {
        "type": "obj",
        "bbox": [620, 780, 940, 980],
        "desc": "A scatter of wood shavings and sawdust across the shop floor in the immediate foreground, curled pale ribbons of planed wood catching warm light, slightly blurred at the bottom edge.",
        "color_palette": ["#E8D5B0", "#C9A36B", "#8C7355"]
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "high_level_description": "A whimsical low-light photograph of an anthropomorphic cat whose face and body are split vertically into a black half and an orange tabby half, wearing a blue tophat and holding a yarn-filled martini glass in one paw and a movie DVD case in the other, set inside a crowded, neon-lit nightclub where a giant mushroom man dances with a bear.",
  "style_description": {
    "aesthetics": "playful surreal portrait photography, sharp focus on the subject with a softly blurred background, shallow depth of field, vivid saturated color",
    "lighting": "moody nightclub ambiance with colorful neon key light, magenta and cyan rim accents, a warm spotlight catching the cat from front-left",
    "photo": "DSLR portrait, 50mm lens, f/2.0, slight motion blur in the background crowd",
    "medium": "photography",
    "color_palette": ["#101012", "#E8741C", "#2E63C9", "#E84B8A", "#3FB55C", "#C9A227"]
  },
  "compositional_deconstruction": {
    "background": "Interior of a busy, dimly lit nightclub seen at night, packed with a blurred dancing crowd, hazy atmosphere thick with colored light beams, glowing magenta and cyan neon signage along the walls, a distant illuminated bar and DJ glow, reflective dark floor scattering specular highlights into bokeh.",
    "elements": [
      {
        "type": "obj",
        "bbox": [110, 180, 400, 760],
        "desc": "A giant mushroom man towering in the midground left, with a broad spotted red-and-cream cap for a head and a plump pale stalk body, arms thrown up mid-dance, slightly motion-blurred and softened by depth of field, lit by shifting neon so its cap glows magenta and cyan at the edges.",
        "color_palette": ["#D24B4B", "#EDE3CF", "#B0457F", "#3A8FC4"]
      },
      {
        "type": "obj",
        "bbox": [360, 240, 650, 780],
        "desc": "A large shaggy brown bear standing on its hind legs in the midground right of center, dancing face-to-face with the mushroom man, one paw raised, fur catching cyan rim light, blurred by the shallow focus to read as a joyful background partner.",
        "color_palette": ["#5A3A22", "#7A4B2A", "#2E8FB0", "#1B1A1C"]
      },
      {
        "type": "obj",
        "bbox": [290, 250, 720, 950],
        "desc": "The main subject, an upright anthropomorphic cat filling the foreground center, its entire body split cleanly down the vertical midline: the left half solid glossy black, the right half warm orange tabby with darker stripes, the seam running straight through nose, chest and belly. It sits tall and forward-facing, both front paws raised to hold props, eyes catching the front spotlight with bright reflective pupils, fur crisply in focus against the soft crowd behind.",
        "color_palette": ["#141414", "#E8741C", "#B85416", "#F0E2C8"]
      },
      {
        "type": "obj",
        "bbox": [395, 175, 625, 360],
        "desc": "A tall royal-blue tophat perched on the cat's head in the upper-center of the frame, with a satin sheen, a slim darker band around the base and a stiff curled brim, neon highlights skating across its smooth surface.",
        "color_palette": ["#2E63C9", "#1A3C8A", "#5C8BE0", "#0E1F4A"]
      },
      {
        "type": "obj",
        "bbox": [225, 540, 430, 825],
        "desc": "A clear martini glass gripped in the cat's right paw (viewer's left, lower-left of frame), tilted slightly, holding a round ball of bright pink yarn instead of a drink, with two slender green knitting needles poking up and out at angles from the yarn; glass rim and stem flash sharp neon specular highlights.",
        "color_palette": ["#E84B8A", "#3FB55C", "#D8E4EC", "#A8285F"]
      },
      {
        "type": "obj",
        "bbox": [615, 560, 835, 835],
        "desc": "A glossy DVD case held flat-faced toward the camera in the cat's left paw (viewer's right, lower-right of frame), its cover showing a standing golden robot rendered in metallic sheen and the bold title text 'This is a test' across the top, plastic surface reflecting magenta and cyan club light.",
        "color_palette": ["#C9A227", "#E8C95A", "#1B1B22", "#E8741C"]
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "high_level_description": "A man stands in a relaxed pose holding up a plain handheld sign reading 'this is a sign' in clear hand-lettered text, photographed in a bright neutral studio setting with a candid, mildly amused mood.",
  "style_description": {
    "aesthetics": "clean editorial portrait photography, sharp focus on the subject, gentle background falloff",
    "lighting": "soft diffused key from the upper left with a subtle fill, neutral-to-slightly-warm color temperature, gentle shadows",
    "photo": "85mm portrait lens, f/4, full-frame DSLR, slight depth of field separating subject from backdrop",
    "medium": "photography",
    "color_palette": ["#E8E2D8", "#C9B79C", "#4A4038", "#8A6F52", "#F5F2EC"]
  },
  "compositional_deconstruction": {
    "background": "A smooth, evenly lit seamless studio backdrop in a warm off-white tone, fading to a marginally darker gradient toward the edges, with soft out-of-focus texture and no distinct objects, creating shallow depth behind the subject.",
    "elements": [
      {
        "type": "obj",
        "bbox": [300, 180, 720, 1000],
        "desc": "A man of average build standing centered-slightly-left and facing the camera, occupying most of the vertical frame from mid-thigh upward, wearing a casual crew-neck sweater and simple trousers; his left arm is raised to grip the bottom of the sign while his right hand steadies its side, head tilted with a faint knowing half-smile and direct eye contact toward the viewer, softly lit from the upper left with a clean catchlight in his eyes.",
        "color_palette": ["#5A4B3A", "#9C8266", "#E2D4C2", "#3A3128"]
      },
      {
        "type": "obj",
        "bbox": [360, 250, 700, 470],
        "desc": "A rectangular flat handheld sign held up at chest-to-shoulder height in the upper-middle of the frame, tilted just slightly off-level, made of plain matte white poster board with a thin neutral border; across its face in bold, evenly spaced hand-lettered black marker reads the phrase 'this is a sign', legible and centered, catching the soft key light with a faint even sheen and casting a gentle shadow onto the man's chest.",
        "color_palette": ["#F7F4EE", "#1E1B17", "#C8C0B2"]
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "high_level_description": "A grizzled bulldog clad in a battered leather jacket stands guard with a sawed-off shotgun beside his motorcycle in a sun-scorched post-apocalyptic desert at harsh late afternoon, exuding a gritty, defiant survivor mood.",
  "style_description": {
    "aesthetics": "moody cinematic, gritty post-apocalyptic realism, high detail with weathered textures, shallow depth of field on the subject",
    "lighting": "harsh low-angle late afternoon sun from the right, long shadows, warm golden key with dusty haze and faint cool fill in the shadows",
    "photo": "full-frame DSLR, 50mm lens, f/2.8, slight atmospheric grain",
    "medium": "photography",
    "color_palette": ["#C8923F", "#7A4B2A", "#3A2E22", "#D9C49A", "#5A6B5E", "#2B2520"]
  },
  "compositional_deconstruction": {
    "background": "A vast cracked desert wasteland stretching to a hazy horizon, dunes and dry parched earth underfoot, rusted debris and a skeletal collapsed structure faintly visible in the far distance, blown dust and heat shimmer thickening the air, pale washed-out sky tinged amber by airborne particulates and the low sun.",
    "elements": [
      {
        "type": "obj",
        "bbox": [40, 360, 360, 620],
        "desc": "Rusted skeletal remains of a collapsed structure in the far background midground-left, sun-bleached metal girders and corrugated sheeting half-buried in sand, small in the frame to establish scale and ruin, partially obscured by dust haze.",
        "color_palette": ["#8A6A45", "#5A4632", "#B8A079"]
      },
      {
        "type": "obj",
        "bbox": [480, 380, 880, 760],
        "desc": "A heavily weathered chopper-style motorcycle parked at a three-quarter angle on the right side of the frame just behind the dog, chrome dulled and pitted, leather seat cracked, dust-caked tank and rust streaks, kickstand down, long shadow cast leftward across the cracked earth.",
        "color_palette": ["#3A2E22", "#7A5A38", "#9A9088", "#2B2520"]
      },
      {
        "type": "obj",
        "bbox": [250, 300, 640, 920],
        "desc": "A muscular English bulldog standing upright and front-and-center slightly left, broad wrinkled jowls and stocky frame facing the viewer with a determined glaring gaze, wearing a scuffed brown leather jacket with frayed collar and metal studs, warm rim light catching the right edge of his face and shoulders, the dominant subject in sharp focus.",
        "color_palette": ["#C8923F", "#6B4528", "#3A2A1E", "#E0C9A0"]
      },
      {
        "type": "obj",
        "bbox": [360, 540, 700, 780],
        "desc": "A sawed-off double-barrel shotgun gripped across the bulldog's body in the lower midground, weathered steel barrels with surface rust and a worn wooden stock, held at a downward diagonal angle catching a glint of the warm sun on the metal.",
        "color_palette": ["#4A4038", "#6B4A2E", "#8A7460"]
      },
      {
        "type": "obj",
        "bbox": [120, 760, 900, 960],
        "desc": "Foreground cracked desert floor in the lower frame, parched fissured clay and scattered sand with small pebbles, dust kicked up faintly, raking warm light emphasizing the deep crack texture and the long shadows of the subject and motorcycle.",
        "color_palette": ["#D9C49A", "#A8895C", "#6B5A40"]
      }
    ]
  }
}
`,
    },
  ],
  neg: '',
  seed: 42,
  walk_seed: true,
  guidance_scale: 4,
  sample_steps: 30,
  num_frames: 1,
  fps: 1,
}