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
  "aspect_ratio": "1:1",
  "high_level_description": "A 35mm film photograph of a red-haired woman in a green jacket playing chess at an outdoor park table, mid-move over a wooden board, while a fiery explosion erupts from a building in the distant background.",
  "compositional_deconstruction": {
    "background": "An urban public park on an overcast afternoon under a pale grey-blue sky, cool-neutral white balance. A grassy lawn with scattered fallen leaves stretches behind the foreground table, bordered by a paved walking path and a row of bare-branched trees. In the far distance, a multi-story stone building erupts in a large orange-and-yellow fireball with a thick black smoke plume rising and rolling outward, sending a faint haze across the upper sky. The blast is out of focus and far off, framed between the tree trunks.",
    "elements": [
      {
        "type": "obj",
        "bbox": [
          180,
          90,
          760,
          520
        ],
        "desc": "Woman seated at a park chess table, leaning slightly forward mid-move. Long wavy red hair falling past her shoulders, fair skin with light freckles, focused expression looking down at the board. Olive-green canvas jacket over a cream knit top, dark jeans. Right hand reaching toward a chess piece, left hand resting on the table edge."
      },
      {
        "type": "obj",
        "bbox": [
          520,
          300,
          830,
          720
        ],
        "desc": "Square wooden chessboard on a round concrete park table, set with a full arrangement of carved boxwood chess pieces in natural and dark-stained wood. A few captured pieces sit off to the side near the board's right edge, one black knight tipped on its side."
      },
      {
        "type": "obj",
        "bbox": [
          300,
          540,
          720,
          760
        ],
        "desc": "Empty green-painted metal park chair with a slatted backrest on the far side of the chess table, facing the woman, slightly angled to the right."
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "aspect_ratio": "1:1",
  "high_level_description": "A 35mm film photograph of a woman in a grey beanie holding a coffee cup while sitting at a wooden cafe table by a window, with a blurred cafe interior behind her.",
  "compositional_deconstruction": {
    "background": "Interior of a small cafe shot in natural daylight with cool-neutral white balance. A large window occupies the left portion of the frame, soft diffused daylight falling across the scene. Exposed brick wall in warm reddish-brown tones runs along the back, partly out of focus. A wooden shelf mounted on the back wall holds a row of white ceramic mugs and a small potted trailing plant. Pendant lights with matte black shades hang from the ceiling, slightly blurred. The floor is wide-plank weathered oak. Distant blurred tables and chairs recede into the soft-focus background on the right side.",
    "elements": [
      {
        "type": "obj",
        "bbox": [
          180,
          300,
          820,
          720
        ],
        "desc": "Woman sitting at a cafe table, facing slightly left toward the window. Light-medium skin tone, shoulder-length wavy auburn hair tucked under a ribbed grey wool beanie. Wearing a cream chunky-knit sweater with sleeves pushed to the forearms. Both hands wrapped around a coffee cup held near chest height, relaxed half-smile, gaze directed out the window."
      },
      {
        "type": "obj",
        "bbox": [
          460,
          400,
          640,
          580
        ],
        "desc": "White ceramic cappuccino cup with a thin handle, held in the woman's hands near chest height. Pale foam visible at the rim with a simple leaf latte-art pattern."
      },
      {
        "type": "obj",
        "bbox": [
          700,
          140,
          1000,
          900
        ],
        "desc": "Rectangular wooden cafe table in warm honey-toned oak, occupying the lower foreground. Visible grain along the surface, one rounded corner facing the camera."
      },
      {
        "type": "obj",
        "bbox": [
          760,
          180,
          940,
          420
        ],
        "desc": "Small folded paper menu card standing upright on the table to the lower left, plain off-white stock with a thin printed border."
      },
      {
        "type": "text",
        "bbox": [
          800,
          210,
          910,
          400
        ],
        "text": "MENU",
        "desc": "Single word in small upright serif capitals, dark grey ink, centered on the front of the folded paper menu card on the table."
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "aspect_ratio": "1:1",
  "high_level_description": "A fish-eye lens photograph of a horse DJing behind turntables at a packed night club, holding a martini glass, surrounded by laser lights and drifting smoke-machine haze on a glowing dance floor.",
  "compositional_deconstruction": {
    "background": "Interior of a dark night club shot through a fish-eye lens with strong barrel distortion bowing the edges of the frame. Black walls and low ceiling studded with mounted laser-light fixtures throwing crisscrossing green and magenta beams that cut through thick drifting haze from a smoke machine. Ambient lighting is dim with cool magenta and cyan washes pooling across a glossy black dance floor that reflects fragmented colored beams. A blurred simplified crowd of clubgoers fills the mid-distance, hands raised, rendered as dark silhouettes against the colored glow.",
    "elements": [
      {
        "type": "obj",
        "bbox": [
          120,
          250,
          720,
          760
        ],
        "desc": "A brown horse standing upright behind a DJ booth in the role of a club DJ, head and long muzzle tilted slightly down toward the equipment, dark mane falling along the neck, alert ears pricked forward. One front hoof rests on a turntable while the other holds aloft a martini glass. Wears large black over-ear headphones around the neck and ears."
      },
      {
        "type": "obj",
        "bbox": [
          600,
          180,
          860,
          840
        ],
        "desc": "A black DJ booth console spanning the lower foreground, fitted with two silver turntables flanking a central mixer with glowing knobs, faders and small green and red LED indicators. Front panel faces the viewer, exaggerated and curved by the fish-eye distortion."
      },
      {
        "type": "obj",
        "bbox": [
          300,
          560,
          470,
          690
        ],
        "desc": "A clear martini glass with a thin stem held aloft, containing pale yellow liquid and a single green olive on a cocktail pick, catching small highlights from the colored club lighting."
      },
      {
        "type": "text",
        "bbox": [
          640,
          360,
          720,
          640
        ],
        "text": "NEON\nSTABLE",
        "desc": "Illuminated club logo on the front face of the DJ booth in a bold sans-serif display typeface, glowing magenta, slightly warped by the fish-eye curvature."
      }
    ]
  }
`,
    },
    {
      prompt: `
{
  "aspect_ratio": "1:1",
  "high_level_description": "A 35mm film photograph of a smiling man proudly showing off his graphic t-shirt on a sandy beach, with a great white shark leaping out of the ocean in the background.",
  "compositional_deconstruction": {
    "background": "Sandy beach scene under a bright overcast sky with cool-neutral white balance. Pale tan sand stretches across the lower portion, slightly damp and packed near the waterline with scattered footprints. Behind the man, the open ocean fills the midground, deep blue-green with choppy whitecaps and rolling waves breaking toward the shore. The horizon line sits high in the frame where the sea meets a hazy pale sky with thin diffuse clouds. Soft even daylight, no harsh shadows, accurate natural color.",
    "elements": [
      {
        "type": "obj",
        "bbox": [
          180,
          540,
          720,
          860
        ],
        "desc": "Great white shark mid-leap, fully breaching the ocean surface in the background, body angled diagonally with mouth open and rows of teeth visible. Grey dorsal surface, white underbelly, water cascading and spraying off its body. Smaller in scale due to distance behind the man."
      },
      {
        "type": "obj",
        "bbox": [
          150,
          260,
          950,
          640
        ],
        "desc": "Man standing on the beach facing the camera, medium-tall build, light-medium skin tone, short brown hair. Grinning widely with a proud expression, gripping the hem of his t-shirt with both hands and pulling it outward to display the front print. Wearing teal swim shorts. Slightly off-center to the left."
      },
      {
        "type": "text",
        "bbox": [
          360,
          330,
          540,
          560
        ],
        "text": "BEACH\nVIBES",
        "desc": "Bold sans-serif print across the chest of the man's white t-shirt, stacked on two lines in navy blue, slightly curved with the fabric as he stretches it toward the camera."
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "aspect_ratio": "1:1",
  "high_level_description": "A brown grizzly bear standing upright on its hind legs, lifting a wooden log onto a half-built log cabin in a snow-covered mountain clearing, with snowy pine forest and peaks behind, rendered as a 35mm film photograph.",
  "compositional_deconstruction": {
    "background": "Snow-covered alpine clearing under a pale overcast winter sky with soft diffused daylight and cool-neutral white balance. Thick fresh snow blankets the ground, undisturbed except around the build site. A dense forest of snow-laden evergreen pines fills the midground, their branches drooping under powder. Jagged grey-and-white granite mountain peaks rise across the distant horizon, partly veiled in light haze. Faint snowflakes drift through the air. The light is even and shadowless across the scene.",
    "elements": [
      {
        "type": "obj",
        "bbox": [
          180,
          120,
          760,
          560
        ],
        "desc": "Large brown grizzly bear standing upright on its hind legs, thick shaggy fur with darker brown legs and a lighter tan muzzle. Front paws gripping a debarked pine log, raising it toward the cabin wall. Head turned in profile, small rounded ears, focused expression, breath fogging in the cold air."
      },
      {
        "type": "obj",
        "bbox": [
          480,
          520,
          880,
          940
        ],
        "desc": "Half-built log cabin made of stacked horizontal debarked pine logs notched and interlocked at the corners. Roughly four log courses high with an open doorway gap on the front face, snow dusting the topmost logs and a small pile of unused logs leaning against the right wall."
      },
      {
        "type": "obj",
        "bbox": [
          700,
          60,
          880,
          320
        ],
        "desc": "Loose stack of cut pine logs lying on the snowy ground in the lower-left foreground, debarked pale tan wood with sawn ends, partially dusted with fresh snow, ready for building."
      },
      {
        "type": "obj",
        "bbox": [
          600,
          400,
          760,
          520
        ],
        "desc": "Rusted double-bit felling axe with a worn wooden handle stuck blade-first into a flat-topped tree stump near the bear's feet, snow gathered on the stump's top surface."
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "aspect_ratio": "1:1",
  "high_level_description": "A punk rocker woman mid-performance on a concert stage, playing an electric guitar and singing into a microphone, with laser lights cutting through haze in a 35mm concert photograph.",
  "compositional_deconstruction": {
    "background": "A dark concert stage shell with a black back wall and exposed steel truss rigging overhead holding stage fixtures. Green and magenta laser beams fan out across the upper space, cutting through a light haze that fills the air and scatters the beams into visible shafts. The stage floor is matte black with scuffed gaffer-tape marks. Distant blurred crowd silhouettes fill the lower foreground edge, lit faintly by stage spill. 35mm concert photograph with cool-neutral white balance and deep shadow contrast.",
    "elements": [
      {
        "type": "obj",
        "bbox": [
          180,
          280,
          860,
          720
        ],
        "desc": "Punk rocker woman standing center stage mid-song, pale skin, spiked bleached-blonde hair with shaved sides, dark smudged eyeliner, mouth open singing with intense expression. Black sleeveless band tee, studded leather choker, ripped black skinny jeans, fingerless gloves. Right hand strumming, left hand on the fretboard, body leaning forward toward the mic."
      },
      {
        "type": "obj",
        "bbox": [
          420,
          200,
          820,
          560
        ],
        "desc": "Black electric guitar with a glossy solid body, white pickguard, chrome hardware and visible strings, slung low across the woman's torso on a studded leather strap, neck angled up toward the upper left."
      },
      {
        "type": "obj",
        "bbox": [
          300,
          560,
          520,
          660
        ],
        "desc": "Black wired stage microphone on a slim chrome boom stand, positioned directly in front of the woman's open mouth, mesh head catching a small highlight from the stage light."
      },
      {
        "type": "obj",
        "bbox": [
          760,
          40,
          980,
          300
        ],
        "desc": "Black foldback stage monitor wedge angled up toward the performer, sitting on the front edge of the stage floor, scuffed casing with a metal grille front."
      },
      {
        "type": "text",
        "bbox": [
          600,
          720,
          720,
          940
        ],
        "text": "RIOT",
        "desc": "Bold uppercase condensed sans-serif band logo in white spray-paint style stenciled across the front of the black speaker stack at lower right, slightly distressed edges."
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "aspect_ratio": "1:1",
  "high_level_description": "A 35mm film photograph of a bearded hipster man assembling a wooden chair on a workbench in a cluttered woodworking shop, surrounded by hand tools and lumber.",
  "compositional_deconstruction": {
    "background": "Interior of a small woodworking workshop with weathered exposed-brick walls on the left and unfinished plywood-panel walls on the right. Sawdust-dusted concrete floor. A pegboard mounted on the rear wall holds rows of hanging hand tools. A single industrial window high on the left wall lets in diffused overcast daylight with a cool-neutral white balance. Fine sawdust haze drifts in the air. Coils of wood shavings and scattered offcuts rest near the wall base. Shot on 35mm film.",
    "elements": [
      {
        "type": "obj",
        "bbox": [
          150,
          300,
          720,
          680
        ],
        "desc": "Bearded hipster man in his mid-thirties, medium-fair skin, full reddish-brown beard and short slicked-back dark hair. Wearing a rolled-sleeve olive flannel shirt, brown leather apron, and dark jeans. Leaning forward over the bench, both hands gripping a wooden chair leg, focused downward expression."
      },
      {
        "type": "obj",
        "bbox": [
          420,
          250,
          820,
          760
        ],
        "desc": "Partially assembled wooden chair made of light oak, seat and two back slats attached, one rear leg detached and held in the man's hands. Raw unfinished surface with visible grain, clamped at one joint with a small metal bar clamp."
      },
      {
        "type": "obj",
        "bbox": [
          600,
          80,
          900,
          920
        ],
        "desc": "Heavy wooden workbench with a thick scarred top, vise mounted on the front-left edge. Surface cluttered with a hand plane, two chisels, a wooden mallet, and a coiled tape measure scattered across the right side."
      },
      {
        "type": "obj",
        "bbox": [
          640,
          40,
          860,
          200
        ],
        "desc": "Cordless drill resting on its side on the bench top near the front-left corner, black and orange body with a chuck-mounted bit."
      },
      {
        "type": "text",
        "bbox": [
          700,
          520,
          760,
          700
        ],
        "text": "OAKWELL\nWORKS",
        "desc": "Small stamped logo branded into the leather apron's chest panel, two stacked lines in a condensed serif font, dark burnt-brown tone on tan leather."
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "aspect_ratio": "1:1",
  "high_level_description": "A studio fashion photograph of a man in a medium shot modeling a casual outfit against a seamless white backdrop, lit with even studio lighting.",
  "compositional_deconstruction": {
    "background": "Seamless white studio backdrop, smoothly lit with even diffused studio lighting from soft boxes on both sides, producing a clean bright cyclorama with no visible seams, corners, or shadows behind the subject. Neutral white balance.",
    "elements": [
      {
        "type": "obj",
        "bbox": [
          80,
          300,
          950,
          720
        ],
        "desc": "Man standing in a medium shot, facing the camera at a slight three-quarter angle. Short dark brown hair neatly styled, light-medium skin tone, clean-shaven, calm neutral expression with a soft closed-mouth look directed at the camera. Wears a fitted heather-grey crewneck t-shirt and dark navy slim chino trousers. Arms relaxed at his sides, shoulders squared. Off-center to the left following rule-of-thirds framing."
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "aspect_ratio": "1:1",
  "high_level_description": "A 35mm film photograph of a man standing on a city sidewalk holding a white cardboard sign reading 'this is a sign', shot at eye-level with neutral daylight.",
  "compositional_deconstruction": {
    "background": "An urban sidewalk scene under overcast daylight with cool-neutral white balance. A grey concrete pavement runs along the bottom, bordered by the brick facade of a low storefront building with large plate-glass windows. A few out-of-focus pedestrians and a parked dark sedan sit in the blurred mid-distance. Pale grey sky visible above the rooflines.",
    "elements": [
      {
        "type": "obj",
        "bbox": [
          180,
          330,
          880,
          680
        ],
        "desc": "Man standing facing the camera, medium build, light skin tone, short brown hair and a trimmed beard. Wearing a charcoal-grey crew-neck shirt and dark blue jeans, relaxed neutral expression looking toward the lens. Both hands raised at chest height gripping the top edge of a cardboard sign held in front of his torso."
      },
      {
        "type": "obj",
        "bbox": [
          400,
          360,
          640,
          660
        ],
        "desc": "Rectangular white cardboard sign with slightly worn edges, held upright in front of the man's chest, plain matte surface with hand-written black marker lettering across the center."
      },
      {
        "type": "text",
        "bbox": [
          460,
          380,
          580,
          640
        ],
        "text": "this is a sign",
        "desc": "Hand-written black marker lettering in a casual sans-serif lowercase style, single line centered across the white cardboard sign."
      }
    ]
  }
}
`,
    },
    {
      prompt: `
{
  "aspect_ratio": "1:1",
  "high_level_description": "A 35mm film photograph of a muscular bulldog in a worn leather jacket standing beside a battered motorcycle in a post-apocalyptic desert, gripping a sawed-off shotgun, with a hazy ruined skyline on the horizon.",
  "compositional_deconstruction": {
    "background": "A sun-scorched post-apocalyptic desert under a pale dust-choked sky, cool-neutral white balance with a thin haze of airborne grit softening the light. Cracked sandy hardpan stretches to a distant horizon where the silhouettes of half-collapsed buildings, a leaning radio tower, and rusted girders rise out of the heat shimmer. Scattered scrub brush and faint tire tracks mark the packed dirt, and a thin band of overcast cloud sits low over the ruined skyline.",
    "elements": [
      {
        "type": "obj",
        "bbox": [
          280,
          330,
          820,
          720
        ],
        "desc": "Muscular English bulldog standing upright on its hind legs in a confident pose, fawn-and-white coat, broad wrinkled face with an underbite and alert dark eyes. Wears a scuffed brown leather biker jacket with a popped collar, frayed cuffs, and a worn metal zipper. Front paws grip a sawed-off double-barrel shotgun held across the chest."
      },
      {
        "type": "obj",
        "bbox": [
          480,
          40,
          880,
          420
        ],
        "desc": "Battered chopper-style motorcycle parked at an angle just left of the bulldog, matte-black fuel tank with chipped paint, rusted chrome exhaust pipes, cracked leather seat, and dusty spoked wheels. Handlebars wrapped in worn tape, a small dented headlamp at the front."
      },
      {
        "type": "obj",
        "bbox": [
          760,
          300,
          900,
          760
        ],
        "desc": "Scattered debris on the desert floor in front of the bulldog: a crushed metal fuel can, a few spent brass shotgun shells, and a broken length of rusted pipe half-buried in the sand."
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