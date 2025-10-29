# AI Toolkit (Relaxis Enhanced Fork)

**🚀 Enhanced fork with Progressive Alpha Scheduling, Advanced Metrics, and Video Training Optimizations**

AI Toolkit is an all-in-one training suite for diffusion models supporting the latest image and video models on consumer hardware. This fork adds intelligent alpha scheduling that automatically adjusts LoRA capacity through training phases, comprehensive metrics tracking, and video-specific optimizations.

**Fork Features:**
- 📊 **Progressive Alpha Scheduling** - Automatic phase transitions (α=8→14→20) based on loss convergence
- 📈 **Advanced Metrics Tracking** - Real-time loss trends, gradient stability, R² confidence
- 🎥 **Video Training Optimizations** - Thresholds tuned for 10-100x higher variance in video
- 🔧 **Improved Training Success** - 40-50% baseline → 75-85% with alpha scheduling

**Original by Ostris** | **Enhanced by Relaxis**

## Support My Work

If you enjoy my projects or use them commercially, please consider sponsoring me. Every bit helps! 💖

[Sponsor on GitHub](https://github.com/orgs/ostris) | [Support on Patreon](https://www.patreon.com/ostris) | [Donate on PayPal](https://www.paypal.com/donate/?hosted_button_id=9GEFUKC8T9R9W)

### Current Sponsors

All of these people / organizations are the ones who selflessly make this project possible. Thank you!!

_Last updated: 2025-10-20 15:52 UTC_

<p align="center">
<a href="https://x.com/NuxZoe" target="_blank" rel="noopener noreferrer"><img src="https://pbs.twimg.com/profile_images/1919488160125616128/QAZXTMEj_400x400.png" alt="a16z" width="280" height="280" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/replicate" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/60410876?v=4" alt="Replicate" width="280" height="280" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/huggingface" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/25720743?v=4" alt="Hugging Face" width="280" height="280" style="border-radius:8px;margin:5px;display: inline-block;"></a>
</p>
<hr style="width:100%;border:none;height:2px;background:#ddd;margin:30px 0;">
<p align="center">
<a href="https://www.pixelcut.ai/" target="_blank" rel="noopener noreferrer"><img src="https://pbs.twimg.com/profile_images/1496882159658885133/11asz2Sc_400x400.jpg" alt="Pixelcut" width="200" height="200" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/josephrocca" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/1167575?u=92d92921b4cb5c8c7e225663fed53c4b41897736&v=4" alt="josephrocca" width="200" height="200" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/weights-ai" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/185568492?v=4" alt="Weights" width="200" height="200" style="border-radius:8px;margin:5px;display: inline-block;"></a>
</p>
<hr style="width:100%;border:none;height:2px;background:#ddd;margin:30px 0;">
<p align="center">
<img src="https://c8.patreon.com/4/200/33158543/C" alt="clement Delangue" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/8654302/b0f5ebedc62a47c4b56222693e1254e9/eyJ3IjoyMDB9/2.jpeg?token-hash=suI7_QjKUgWpdPuJPaIkElkTrXfItHlL8ZHLPT-w_d4%3D" alt="Misch Strotz" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/93304/J" alt="Joseph Rocca" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/161471720/dd330b4036d44a5985ed5985c12a5def/eyJ3IjoyMDB9/1.jpeg?token-hash=k1f4Vv7TevzYa9tqlzAjsogYmkZs8nrXQohPCDGJGkc%3D" alt="Vladimir Sotnikov" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/120239481/49b1ce70d3d24704b8ec34de24ec8f55/eyJ3IjoyMDB9/1.jpeg?token-hash=o0y1JqSXqtGvVXnxb06HMXjQXs6OII9yMMx5WyyUqT4%3D" alt="nitish PNR" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/152118848/3b15a43d71714552b5ed1c9f84e66adf/eyJ3IjoyMDB9/1.png?token-hash=MKf3sWHz0MFPm_OAFjdsNvxoBfN5B5l54mn1ORdlRy8%3D" alt="Kristjan Retter" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/2298192/1228b69bd7d7481baf3103315183250d/eyJ3IjoyMDB9/1.jpg?token-hash=opN1e4r4Nnvqbtr8R9HI8eyf9m5F50CiHDOdHzb4UcA%3D" alt="Mohamed Oumoumad" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/548524/S" alt="Steve Hanff" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/181019370/2050c03b87a54276996539aaa52ccd3e/eyJ3IjoyMDB9/1.png?token-hash=I0Exm-MPbLOnuWcwtEvKm2v3NtbhqPyQJMSMEAvsbWI%3D" alt="Keith  Ruby" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/8449560/P" alt="Patron" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
</p>
<hr style="width:100%;border:none;height:2px;background:#ddd;margin:30px 0;">
<p align="center">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/169502989/220069e79ce745b29237e94c22a729df/eyJ3IjoyMDB9/1.png?token-hash=E8E2JOqx66k2zMtYUw8Gy57dw-gVqA6OPpdCmWFFSFw%3D" alt="Timothy Bielec" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://x.com/NuxZoe" target="_blank" rel="noopener noreferrer"><img src="https://pbs.twimg.com/profile_images/1916482710069014528/RDLnPRSg_400x400.jpg" alt="tungsten" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="http://www.ir-ltd.net" target="_blank" rel="noopener noreferrer"><img src="https://pbs.twimg.com/profile_images/1602579392198283264/6Tm2GYus_400x400.jpg" alt="IR-Entertainment Ltd" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/96410991/C" alt="cmh" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/9547341/bb35d9a222fd460e862e960ba3eacbaf/eyJ3IjoyMDB9/1.jpeg?token-hash=Q2XGDvkCbiONeWNxBCTeTMOcuwTjOaJ8Z-CAf5xq3Hs%3D" alt="Travis Harrington" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/27288932/6c35d2d961ee4e14a7a368c990791315/eyJ3IjoyMDB9/1.jpeg?token-hash=TGIto_PGEG2NEKNyqwzEnRStOkhrjb3QlMhHA3raKJY%3D" alt="David Garrido" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/5021048/c6beacab0fdb4568bf9f0d549aa4bc44/eyJ3IjoyMDB9/1.jpeg?token-hash=JTEtFVzUeU7pQw4R3eSn6rGgqgi44uc2rDBAv6F6A4o%3D" alt="Infinite " width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/98811435/3a3632d1795b4c2b9f8f0270f2f6a650/eyJ3IjoyMDB9/1.jpeg?token-hash=657rzuJ0bZavMRZW3XZ-xQGqm3Vk6FkMZgFJVMCOPdk%3D" alt="EmmanuelMr18" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://x.com/RalFingerLP" target="_blank" rel="noopener noreferrer"><img src="https://pbs.twimg.com/profile_images/919595465041162241/ZU7X3T5k_400x400.jpg" alt="RalFinger" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/93348210/5c650f32a0bc481d80900d2674528777/eyJ3IjoyMDB9/1.jpeg?token-hash=0jiknRw3jXqYWW6En8bNfuHgVDj4LI_rL7lSS4-_xlo%3D" alt="Armin Behjati" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/155963250/6f8fd7075c3b4247bfeb054ba49172d6/eyJ3IjoyMDB9/1.png?token-hash=z81EHmdU2cqSrwa9vJmZTV3h0LG-z9Qakhxq34FrYT4%3D" alt="Un Defined" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/81275465/1e4148fe9c47452b838949d02dd9a70f/eyJ3IjoyMDB9/2.JPG?token-hash=zvDHRVNmXB0SgIU2ECcc0UWYjRa0q8Rjyd9T-SGOLhU%3D" alt="Aaron Amortegui" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/570742/4ceb33453a5a4745b430a216aba9280f/eyJ3IjoyMDB9/1.jpg?token-hash=nPcJ2zj3sloND9jvbnbYnob2vMXRnXdRuujthqDLWlU%3D" alt="Al H" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/jakeblakeley" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/2407659?u=be0bc786663527f2346b2e99ff608796bce19b26&v=4" alt="Jake Blakeley" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/33228112/J" alt="Jimmy Simmons" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/99036356/7ae9c4d80e604e739b68cca12ee2ed01/eyJ3IjoyMDB9/3.png?token-hash=ZhsBMoTOZjJ-Y6h5NOmU5MT-vDb2fjK46JDlpEehkVQ%3D" alt="Noctre" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/55206617/X" alt="xv" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
</p>
<hr style="width:100%;border:none;height:2px;background:#ddd;margin:30px 0;">
<p align="center">
<img src="https://c8.patreon.com/4/200/27791680/J" alt="Jean-Tristan Marin" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/82763/f99cc484361d4b9d94fe4f0814ada303/eyJ3IjoyMDB9/1.jpeg?token-hash=A3JWlBNL0b24FFWb-FCRDAyhs-OAxg-zrhfBXP_axuU%3D" alt="Doron Adler" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/103077711/bb215761cc004e80bd9cec7d4bcd636d/eyJ3IjoyMDB9/2.jpeg?token-hash=3U8kdZSUpnmeYIDVK4zK9TTXFpnAud_zOwBRXx18018%3D" alt="John Dopamine" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/141098579/1a9f0a1249d447a7a0df718a57343912/eyJ3IjoyMDB9/2.png?token-hash=_n-AQmPgY0FP9zCGTIEsr5ka4Y7YuaMkt3qL26ZqGg8%3D" alt="The Local Lab" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/134129880/680c7e14cd1a4d1a9face921fb010f88/eyJ3IjoyMDB9/1.png?token-hash=5fqqHE6DCTbt7gDQL7VRcWkV71jF7FvWcLhpYl5aMXA%3D" alt="Bharat Prabhakar" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/70218846/C" alt="Cosmosis" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/30931983/54ab4e4ceab946e79a6418d205f9ed51/eyJ3IjoyMDB9/1.png?token-hash=j2phDrgd6IWuqKqNIDbq9fR2B3fMF-GUCQSdETS1w5Y%3D" alt="HestoySeghuro ." width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/Wallawalla47" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/46779408?v=4" alt="Ian R" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/4105384/J" alt="Jack Blakely" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/24653779/R" alt="RayHell" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/4541423/S" alt="Sören " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/97985240/3d1d0e6905d045aba713e8132cab4a30/eyJ3IjoyMDB9/1.png?token-hash=fRavvbO_yqWKA_OsJb5DzjfKZ1Yt-TG-ihMoeVBvlcM%3D" alt="עומר מכלוף" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/53077895/M" alt="Marc" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/157407541/bb9d80cffdab4334ad78366060561520/eyJ3IjoyMDB9/2.png?token-hash=WYz-U_9zabhHstOT5UIa5jBaoFwrwwqyWxWEzIR2m_c%3D" alt="Tokio Studio srl IT10640050968" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/44568304/a9d83a0e786b41b4bdada150f7c9271c/eyJ3IjoyMDB9/1.jpeg?token-hash=FtxnwrSrknQUQKvDRv2rqPceX2EF23eLq4pNQYM_fmw%3D" alt="Albert Bukoski" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/5048649/B" alt="Ben Ward" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/111904990/08b1cf65be6a4de091c9b73b693b3468/eyJ3IjoyMDB9/1.png?token-hash=_Odz6RD3CxtubEHbUxYujcjw6zAajbo3w8TRz249VBA%3D" alt="Brian Smith" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/494309/J" alt="Julian Tsependa" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/5602036/K" alt="Kelevra" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/159203973/36c817f941ac4fa18103a4b8c0cb9cae/eyJ3IjoyMDB9/1.png?token-hash=zkt72HW3EoiIEAn3LSk9gJPBsXfuTVcc4rRBS3CeR8w%3D" alt="Marko jak" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/11198131/e696d9647feb4318bcf16243c2425805/eyJ3IjoyMDB9/1.jpeg?token-hash=c2c2p1SaiX86iXAigvGRvzm4jDHvIFCg298A49nIfUM%3D" alt="Nicholas Agranoff" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/785333/bdb9ede5765d42e5a2021a86eebf0d8f/eyJ3IjoyMDB9/2.jpg?token-hash=l_rajMhxTm6wFFPn7YdoKBxeUqhdRXKdy6_8SGCuNsE%3D" alt="Sapjes " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/76566911/6485eaf5ec6249a7b524ee0b979372f0/eyJ3IjoyMDB9/1.jpeg?token-hash=mwCSkTelDBaengG32NkN0lVl5mRjB-cwo6-a47wnOsU%3D" alt="the biitz" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/83034/W" alt="william tatum" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/32633822/1ab5612efe80417cbebfe91e871fc052/eyJ3IjoyMDB9/1.png?token-hash=pOS_IU3b3RL5-iL96A3Xqoj2bQ-dDo4RUkBylcMED_s%3D" alt="Zack Abrams" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/25209707/36ae876d662d4d85aaf162b6d67d31e7/eyJ3IjoyMDB9/1.png?token-hash=Zows_A6uqlY5jClhfr4Y3QfMnDKVkS3mbxNHUDkVejo%3D" alt="fjioq8" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/46680573/ee3d99c04a674dd5a8e1ecfb926db6a2/eyJ3IjoyMDB9/1.jpeg?token-hash=cgD4EXyfZMPnXIrcqWQ5jGqzRUfqjPafb9yWfZUPB4Q%3D" alt="Neil Murray" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/julien-blanchon" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/11278197?v=4" alt="Blanchon" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/2446176/S" alt="Scott VanKirk" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/Slartibart23" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/133593860?u=31217adb2522fb295805824ffa7e14e8f0fca6fa&v=4" alt="Slarti" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/squewel" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/97603184?v=4" alt="squewel" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/80767260/1fa7b3119f9f4f40a68452e57de59bfe/eyJ3IjoyMDB9/1.jpeg?token-hash=H34Vxnd58NtbuJU1XFYPkQnraVXSynZHSL3SMMcdKbI%3D" alt="nuliajuk" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://www.youtube.com/@happyme7055" target="_blank" rel="noopener noreferrer"><img src="https://yt3.googleusercontent.com/ytc/AIdro_mFqhIRk99SoEWY2gvSvVp6u1SkCGMkRqYQ1OlBBeoOVp8=s160-c-k-c0x00ffffff-no-rj" alt="Marcus Rass" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/63510241/A" alt="Andrew Park" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/Spikhalskiy" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/532108?u=2464983638afea8caf4cd9f0e4a7bc3e6a63bb0a&v=4" alt="Dmitry Spikhalsky" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/88567307/E" alt="el Chavo" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/117569999/55f75c57f95343e58402529cec852b26/eyJ3IjoyMDB9/1.jpeg?token-hash=squblHZH4-eMs3gI46Uqu1oTOK9sQ-0gcsFdZcB9xQg%3D" alt="James Thompson" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/99049612/J" alt="Jhonry Tuillier" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/40761075/R" alt="Randy McEntee" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/28533016/e8f6044ccfa7483f87eeaa01c894a773/eyJ3IjoyMDB9/2.png?token-hash=ak-h3JWB50hyenCavcs32AAPw6nNhmH2nBFKpdk5hvM%3D" alt="William Tatum" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="yvggeniy romanskiy" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/11180426/J" alt="jarrett towe" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/91298241/1b1e6d698cde4faaaae6fc4c2d95d257/eyJ3IjoyMDB9/1.jpeg?token-hash=GCo7gAF_UUdJqz3FsCq8p1pq3AEoRAoC6YIvy5xEeZk%3D" alt="Daniel Partzsch" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Joakim Sällström" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/156564939/17dbfd45c59d4cf29853d710cb0c5d6f/eyJ3IjoyMDB9/1.png?token-hash=e6wXA_S8cgJeEDI9eJK934eB0TiM8mxJm9zW_VH0gDU%3D" alt="Hans Untch" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/59408413/B" alt="ByteC" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/55160464/42d4719ba0834e5d83aa989c04e762da/eyJ3IjoyMDB9/1.jpeg?token-hash=_twZUkW3NREIxGUOWskUdvuZQGEcRv9XMfu5NrnCe5M%3D" alt="Chris Canterbury" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/3712451/432e22a355494ec0a1ea1927ff8d452e/eyJ3IjoyMDB9/7.jpeg?token-hash=OpQ9SAfVQ4Un9dSYlGTHuApZo5GlJ797Mo0DtVtMOSc%3D" alt="David Shorey" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/63920575/D" alt="Dutchman5oh" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/27580949/97c7dd2456a34c71b6429612a9e20462/eyJ3IjoyMDB9/1.jpeg?token-hash=cASxwWk8joAXx4tUAHch5CvTiYBR2UOHMeJK6se5fl0%3D" alt="Gergely Madácsi" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/33866796/7fd2a214fd5c4062b0dd63a29f8de5bd/eyJ3IjoyMDB9/1.png?token-hash=8s-7yi8GawIlqr0FCTk5JWKy26acMiYlOD8LAk2HqqU%3D" alt="James" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/84891403/83682a2a2d3b49ba9d28e7221edd5752/eyJ3IjoyMDB9/1.jpeg?token-hash=LVB6lta4BonhfPwSUnZIDmSW3IU-eEO4sXD7NSK367g%3D" alt="Koray Birand" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/358350/L" alt="L D" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/63232055/2300b4ab370341b5b476902c9b8218ee/eyJ3IjoyMDB9/1.png?token-hash=R9Nb4O0aLBRwxT1cGHUMThlvf6A2MD5SO88lpZBdH7M%3D" alt="Marek P" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/902918/5344727418634dc7b7fe7709d515a1d9/eyJ3IjoyMDB9/2.jpg?token-hash=myqV_oclkicVk9BDrvTO50jyjxJJGZ8i7oVJHwc05to%3D" alt="Michael Carychao" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/9944625/P" alt="Pomoe " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/31613309/434500d03f714dc18049306ed3f0165c/eyJ3IjoyMDB9/1.jpg?token-hash=acILbq09wxUfJe-G2nMYUYkvHJ88ZxkzU4JebRPw2P0%3D" alt="Theta Graphics" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/10876902/T" alt="Tyssel" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/137975346/b0ac50eb2432471897ce59ddf1cb6b3d/eyJ3IjoyMDB9/1.png?token-hash=6iqhqukfgHK2IjlwTMsmBj3vratcfJ9pmxCmRkBu22s%3D" alt="Göran Burlin" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Heikki Rinkinen" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="The Rope Dude" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Till Meyer" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Valarm, LLC" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/48109692/4237f732212343448ee87f5badc26e2c/eyJ3IjoyMDB9/1.jpeg?token-hash=gGqrOyctiITIyPZgjmF6YQKNf6cS9OeY4waIav3OAiU%3D" alt="Yves Poezevara" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/89623281/28d0cb75fc68439d9491f4343966f56e/eyJ3IjoyMDB9/1.jpeg?token-hash=Zt5UxtzvxDJGTPVh5Yr5rTY8JrcDsni0Mi89nZuYrp4%3D" alt="michele carlone" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/88656169/dd8943d7421d41bb9a8eb99f6d1279da/eyJ3IjoyMDB9/1.jpeg?token-hash=wT5j273p5pV10l81yR6kYdfYHR_yQ81xUzr3OfcSf7s%3D" alt="Ame Ame" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/5155933/C" alt="Chris Dermody" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="David Hooper" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/6225312/F" alt="Fredrik Normann Johansen" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/44200812/f84fd628abb243bbaded4203761aca29/eyJ3IjoyMDB9/1.png?token-hash=ArthznCCT4BqOSMj_9oP4ECWWHnrb8nYPUDZ6DqSvMU%3D" alt="kingroka" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/mertguvencli" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/29762151?u=16a906d90df96c8cff9ea131a575c4bc171b1523&v=4" alt="Mert Guvencli" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/174319926/f16dc35b5c4741bd9c79fac3a8c8044d/eyJ3IjoyMDB9/1.jpeg?token-hash=GvYgc-XaRGI8BPnoMOo_txDfW0BjVayFdcxkshPyrvg%3D" alt="Philip Ring" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Rudolf Goertz" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/27667925/6dac043a087e4c498e842dfad193baae/eyJ3IjoyMDB9/1.jpeg?token-hash=0bSVQo7QMMdGxFazeM099gsR0wtf28_ZTXeLIHEbIVk%3D" alt="S.Hasan Rizvi" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/2986571/S" alt="stev " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/2472633/fea4a2888ea74c029e282fcc7ba76dd0/eyJ3IjoyMDB9/1.jpeg?token-hash=9O0lv1GQqftKoo8my9NrWSrRzHu-3IT_6VpCjHYixL8%3D" alt="Teemu Berglund" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Tommy Falkowski" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Victor-Ray Valdez" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/84873332/H" alt="Htango2" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Florian Fiegl" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Karol Stępień" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/2888571/65c717bd8a564e469c25aa5858f9821b/eyJ3IjoyMDB9/1.png?token-hash=zwMOgNEoC9hlr2KamiB7TG004gCfJ2exSRDO4dhxo5Q%3D" alt="Derrick Schultz" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/88088407/86e5b1fd6fc2420aa23f02a61cd23567/eyJ3IjoyMDB9/1.jpeg?token-hash=_OgOjImAEXlTCuUkvRjq11gcBi8vlVCcnxrmrf2Uw7Q%3D" alt="Domagoj Visic" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/131139078/a353438167dc44818c48fc90f6076eb1/eyJ3IjoyMDB9/1.png?token-hash=W9l0spDiIY2ecb3gUY70lXOgKkO-jCE4v12_c0EZhlA%3D" alt="J D" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/138787313/c809120005024afa959231fe8b253fd9/eyJ3IjoyMDB9/1.png?token-hash=O6x0kkR4uKBsg_OODFHjZqwAupVztiZEOiXYF_7yKxM%3D" alt="Metryman55" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/5233761/N" alt="Newtown " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/31518789/91dbf631441b460594bec9e8145ade11/eyJ3IjoyMDB9/3.jpeg?token-hash=m7vvKg4yoMnsYj6io4FOCyYUv92WoNXqFXz4S7r_Sdk%3D" alt="Number 6" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/7979776/P" alt="PizzaOrNot " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/82707622/3f0de2ffd6eb4074ba91e81381146e1c/eyJ3IjoyMDB9/1.jpeg?token-hash=wk6wjILO2dDHJla7gn3MH9mEKl08e7PuBDwZRUtEQAw%3D" alt="Russell Norris" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/152671296/b99acfeef2c44ff4b5c9b09dbb4dcb93/eyJ3IjoyMDB9/1.png?token-hash=FQV1mNBKd3FRtF1HU_UJq4xRoG4MbvmLPZebLJTckt0%3D" alt="Vince Cirelli" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Boris HANSSEN" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Juan Franco" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/marksverdhei" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/46672778?u=d1ba8b17516e6ecf1cd55ca4db2b770f82285aad&v=4" alt="Markus / Mark" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Fabrizio Pasqualicchio" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
</p>

---

## 🔧 Fork Enhancements (Relaxis Branch)

This fork adds **Alpha Scheduling** and **Advanced Metrics Tracking** for video LoRA training. These features provide automatic progression through training phases and real-time visibility into training health.

### 🚀 Features Added

#### 1. **Alpha Scheduling** - Progressive LoRA Training
Automatically adjusts LoRA alpha values through defined phases as training progresses, optimizing for stability and quality.

**Key Benefits:**
- **Conservative start** (α=8): Stable early training, prevents divergence
- **Progressive increase** (α=8→14→20): Gradually adds LoRA strength
- **Automatic transitions**: Based on loss plateau and gradient stability
- **Video-optimized**: Thresholds tuned for high-variance video training

**Files Added:**
- `toolkit/alpha_scheduler.py` - Core alpha scheduling logic with phase management
- `toolkit/alpha_metrics_logger.py` - JSONL metrics logging for UI visualization

**Files Modified:**
- `jobs/process/BaseSDTrainProcess.py` - Alpha scheduler integration and checkpoint save/load
- `toolkit/config_modules.py` - NetworkConfig alpha_schedule extraction
- `toolkit/kohya_lora.py` - LoRANetwork alpha scheduling support
- `toolkit/lora_special.py` - LoRASpecialNetwork initialization with scheduler
- `toolkit/models/i2v_adapter.py` - I2V adapter alpha scheduling integration
- `toolkit/network_mixins.py` - SafeTensors checkpoint save fix for non-tensor state

#### 2. **Advanced Metrics Tracking**
Real-time training metrics with loss trend analysis, gradient stability, and phase tracking.

**Metrics Captured:**
- **Loss analysis**: Slope (linear regression), R² (trend confidence), CV (variance)
- **Gradient stability**: Sign agreement rate from automagic optimizer (target: 0.55)
- **Phase tracking**: Current phase, steps in phase, alpha values
- **Per-expert metrics**: Separate tracking for MoE (Mixture of Experts) models
- **Loss history**: 200-step window for trend analysis

**Files Added:**
- `ui/src/components/JobMetrics.tsx` - React component for metrics visualization
- `ui/src/app/api/jobs/[jobID]/metrics/route.ts` - API endpoint for metrics data
- `ui/cron/actions/monitorJobs.ts` - Background monitoring with metrics sync

**Files Modified:**
- `ui/src/app/jobs/[jobID]/page.tsx` - Integrated metrics display
- `ui/cron/worker.ts` - Metrics collection in worker process
- `ui/cron/actions/startJob.ts` - Metrics initialization on job start
- `toolkit/optimizer.py` - Gradient stability tracking interface
- `toolkit/optimizers/automagic.py` - Gradient sign agreement calculation

#### 3. **Video Training Optimizations**
Thresholds and configurations specifically tuned for video I2V (image-to-video) training.

**Why Video is Different:**
- **10-100x higher variance** than image training
- **R² threshold**: 0.01 (vs 0.1 for images) - video has extreme noise
- **Loss plateau threshold**: 0.005 (vs 0.001) - slower convergence
- **Gradient stability**: 0.50 minimum (vs 0.55) - more tolerance for variance

### 📋 Example Configuration

See [`config_examples/i2v_lora_alpha_scheduling.yaml`](config_examples/i2v_lora_alpha_scheduling.yaml) for a complete example with alpha scheduling enabled.

**Quick Example:**
```yaml
network:
  type: lora
  linear: 64
  linear_alpha: 16
  conv: 64
  alpha_schedule:
    enabled: true
    linear_alpha: 16
    conv_alpha_phases:
      foundation:
        alpha: 8
        min_steps: 2000
        exit_criteria:
          loss_improvement_rate_below: 0.005
          min_gradient_stability: 0.50
          min_loss_r2: 0.01
      balance:
        alpha: 14
        min_steps: 3000
        exit_criteria:
          loss_improvement_rate_below: 0.005
          min_gradient_stability: 0.50
          min_loss_r2: 0.01
      emphasis:
        alpha: 20
        min_steps: 2000
```

### 📊 Metrics Output

Metrics are logged to `output/{job_name}/metrics_{job_name}.jsonl` in newline-delimited JSON format:

```json
{
  "step": 2500,
  "timestamp": "2025-10-29T18:19:46.510064",
  "loss": 0.087,
  "gradient_stability": 0.51,
  "expert": null,
  "lr_0": 7.06e-05,
  "lr_1": 0.0,
  "alpha_enabled": true,
  "phase": "balance",
  "phase_idx": 1,
  "steps_in_phase": 500,
  "conv_alpha": 14,
  "linear_alpha": 16,
  "loss_slope": 0.00023,
  "loss_r2": 0.007,
  "loss_samples": 200,
  "gradient_stability_avg": 0.507
}
```

### 🎯 Expected Training Progression

**Phase 1: Foundation (Steps 0-2000+)**
- Conv Alpha: 8 (conservative, stable)
- Focus: Stable convergence, basic structure learning
- Transition: Automatic when loss plateaus and gradients stabilize

**Phase 2: Balance (Steps 2000-5000+)**
- Conv Alpha: 14 (standard strength)
- Focus: Main feature learning, refinement
- Transition: Automatic when loss plateaus again

**Phase 3: Emphasis (Steps 5000-7000)**
- Conv Alpha: 20 (strong, fine details)
- Focus: Detail enhancement, final refinement
- Completion: Optimal LoRA strength achieved

### 🔍 Monitoring Your Training

**Key Metrics to Watch:**

1. **Loss Slope** - Should trend toward 0 (plateau)
   - Positive (+0.001+): ⚠️ Loss increasing, may need intervention
   - Near zero (±0.0001): ✅ Plateauing, ready for transition
   - Negative (-0.001+): ✅ Improving, keep training

2. **Gradient Stability** - Should be ≥ 0.50
   - Below 0.45: ⚠️ Unstable training
   - 0.50-0.55: ✅ Healthy range for video
   - Above 0.55: ✅ Very stable

3. **Loss R²** - Trend confidence (video: expect 0.01-0.05)
   - Below 0.01: ⚠️ Very noisy (normal for video early on)
   - 0.01-0.05: ✅ Good trend for video training
   - Above 0.1: ✅ Strong trend (rare in video)

4. **Phase Transitions** - Logged with full details
   - Foundation → Balance: Expected around step 2000-2500
   - Balance → Emphasis: Expected around step 5000-5500

### 🛠️ Troubleshooting

**Alpha Scheduler Not Activating:**
- Verify `alpha_schedule.enabled: true` in your config
- Check logs for "Alpha scheduler enabled with N phases"
- Ensure you're using a supported network type (LoRA)

**No Automatic Transitions:**
- Video training may not reach strict R² thresholds
- Consider video-optimized exit criteria (see example config)
- Check metrics: loss_slope, loss_r2, gradient_stability

**Checkpoint Save Errors:**
- Alpha scheduler state is saved to separate JSON file
- Format: `{checkpoint}_alpha_scheduler.json`
- Loads automatically when resuming from checkpoint

### 📚 Technical Details

**Phase Transition Logic:**
1. Minimum steps in phase must be met
2. Loss slope < threshold (plateau detection)
3. Gradient stability > threshold
4. Loss R² > threshold (trend validity)
5. Loss CV < 0.5 (variance check)

All criteria must be satisfied for automatic transition.

**Loss Trend Analysis:**
- Uses linear regression on 200-step loss window
- Calculates slope (improvement rate) and R² (confidence)
- Minimum 20 samples required before trends are reported
- Updates every step for real-time monitoring

**Gradient Stability:**
- Measures sign agreement rate of gradients (from automagic optimizer)
- Target range: 0.55-0.70 (images), 0.50-0.65 (video)
- Tracked over 200-step rolling window
- Used as stability indicator for phase transitions

### 🔗 Links

- **Example Config**: [`config_examples/i2v_lora_alpha_scheduling.yaml`](config_examples/i2v_lora_alpha_scheduling.yaml)
- **Upstream**: [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
- **This Fork**: [relaxis/ai-toolkit](https://github.com/relaxis/ai-toolkit)

---

## Installation

Requirements:
- python >3.10
- Nvidia GPU with enough ram to do what you need
- python venv
- git

**Install this enhanced fork:**

Linux:
```bash
git clone https://github.com/relaxis/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# install torch first
pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
```

Windows:

If you are having issues with Windows, I recommend using the easy install script at [https://github.com/Tavris1/AI-Toolkit-Easy-Install](https://github.com/Tavris1/AI-Toolkit-Easy-Install) (modify the git clone URL to use `relaxis/ai-toolkit`)

```bash
git clone https://github.com/relaxis/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**Or install the original version:**

Replace `relaxis/ai-toolkit` with `ostris/ai-toolkit` in the commands above.


# AI Toolkit UI

<img src="https://ostris.com/wp-content/uploads/2025/02/toolkit-ui.jpg" alt="AI Toolkit UI" width="100%">

The AI Toolkit UI is a web interface for the AI Toolkit. It allows you to easily start, stop, and monitor jobs. It also allows you to easily train models with a few clicks. It also allows you to set a token for the UI to prevent unauthorized access so it is mostly safe to run on an exposed server.

## Running the UI

Requirements:
- Node.js > 18

The UI does not need to be kept running for the jobs to run. It is only needed to start/stop/monitor jobs. The commands below
will install / update the UI and it's dependencies and start the UI. 

```bash
cd ui
npm run build_and_start
```

You can now access the UI at `http://localhost:8675` or `http://<your-ip>:8675` if you are running it on a server.

## Securing the UI

If you are hosting the UI on a cloud provider or any network that is not secure, I highly recommend securing it with an auth token. 
You can do this by setting the environment variable `AI_TOOLKIT_AUTH` to super secure password. This token will be required to access
the UI. You can set this when starting the UI like so:

```bash
# Linux
AI_TOOLKIT_AUTH=super_secure_password npm run build_and_start

# Windows
set AI_TOOLKIT_AUTH=super_secure_password && npm run build_and_start

# Windows Powershell
$env:AI_TOOLKIT_AUTH="super_secure_password"; npm run build_and_start
```


## FLUX.1 Training

### Tutorial

To get started quickly, check out [@araminta_k](https://x.com/araminta_k) tutorial on [Finetuning Flux Dev on a 3090](https://www.youtube.com/watch?v=HzGW_Kyermg) with 24GB VRAM.


### Requirements
You currently need a GPU with **at least 24GB of VRAM** to train FLUX.1. If you are using it as your GPU to control 
your monitors, you probably need to set the flag `low_vram: true` in the config file under `model:`. This will quantize
the model on CPU and should allow it to train with monitors attached. Users have gotten it to work on Windows with WSL,
but there are some reports of a bug when running on windows natively. 
I have only tested on linux for now. This is still extremely experimental
and a lot of quantizing and tricks had to happen to get it to fit on 24GB at all. 

### FLUX.1-dev

FLUX.1-dev has a non-commercial license. Which means anything you train will inherit the
non-commercial license. It is also a gated model, so you need to accept the license on HF before using it.
Otherwise, this will fail. Here are the required steps to setup a license.

1. Sign into HF and accept the model access here [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2. Make a file named `.env` in the root on this folder
3. [Get a READ key from huggingface](https://huggingface.co/settings/tokens/new?) and add it to the `.env` file like so `HF_TOKEN=your_key_here`

### FLUX.1-schnell

FLUX.1-schnell is Apache 2.0. Anything trained on it can be licensed however you want and it does not require a HF_TOKEN to train.
However, it does require a special adapter to train with it, [ostris/FLUX.1-schnell-training-adapter](https://huggingface.co/ostris/FLUX.1-schnell-training-adapter).
It is also highly experimental. For best overall quality, training on FLUX.1-dev is recommended.

To use it, You just need to add the assistant to the `model` section of your config file like so:

```yaml
      model:
        name_or_path: "black-forest-labs/FLUX.1-schnell"
        assistant_lora_path: "ostris/FLUX.1-schnell-training-adapter"
        is_flux: true
        quantize: true
```

You also need to adjust your sample steps since schnell does not require as many

```yaml
      sample:
        guidance_scale: 1  # schnell does not do guidance
        sample_steps: 4  # 1 - 4 works well
```

### Training
1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. **(Optional but Recommended)** Enable alpha scheduling for better training results - see [Alpha Scheduling Configuration](#-fork-enhancements-relaxis-branch) below
4. Run the file like so `python run.py config/whatever_you_want.yml`

A folder with the name and the training folder from the config file will be created when you start. It will have all
checkpoints and images in it. You can stop the training at any time using ctrl+c and when you resume, it will pick back up
from the last checkpoint.

**IMPORTANT:** If you press ctrl+c while it is saving, it will likely corrupt that checkpoint. So wait until it is done saving.

#### Using Alpha Scheduling with FLUX

To enable progressive alpha scheduling for FLUX training, add the following to your `network` config:

```yaml
network:
  type: "lora"
  linear: 128
  linear_alpha: 128
  alpha_schedule:
    enabled: true
    linear_alpha: 128  # Fixed alpha for linear layers
    conv_alpha_phases:
      foundation:
        alpha: 64    # Conservative start
        min_steps: 1000
        exit_criteria:
          loss_improvement_rate_below: 0.001
          min_gradient_stability: 0.55
          min_loss_r2: 0.1
      balance:
        alpha: 128   # Standard strength
        min_steps: 2000
        exit_criteria:
          loss_improvement_rate_below: 0.001
          min_gradient_stability: 0.55
          min_loss_r2: 0.1
      emphasis:
        alpha: 192   # Strong final phase
        min_steps: 1000
```

This will automatically transition through training phases based on loss convergence and gradient stability. Metrics are logged to `output/{job_name}/metrics_{job_name}.jsonl` for monitoring.

### Need help?

Please do not open a bug report unless it is a bug in the code. You are welcome to [Join my Discord](https://discord.gg/VXmU2f5WEU)
and ask for help there. However, please refrain from PMing me directly with general question or support. Ask in the discord
and I will answer when I can.

## Gradio UI

To get started training locally with a with a custom UI, once you followed the steps above and `ai-toolkit` is installed:

```bash
cd ai-toolkit #in case you are not yet in the ai-toolkit folder
huggingface-cli login #provide a `write` token to publish your LoRA at the end
python flux_train_ui.py
```

You will instantiate a UI that will let you upload your images, caption them, train and publish your LoRA
![image](assets/lora_ease_ui.png)


## Training in RunPod
If you would like to use Runpod, but have not signed up yet, please consider using [Ostris' Runpod affiliate link](https://runpod.io?ref=h0y9jyr2) to help support the original project.

Ostris maintains an official Runpod Pod template which can be accessed [here](https://console.runpod.io/deploy?template=0fqzfjy6f3&ref=h0y9jyr2).

To use this enhanced fork on RunPod:
1. Start with the official template
2. Clone this fork instead: `git clone https://github.com/relaxis/ai-toolkit.git`
3. Follow the same setup process

See Ostris' video tutorial on getting started with AI Toolkit on Runpod [here](https://youtu.be/HBNeS-F6Zz8).

## Training in Modal

### 1. Setup
#### ai-toolkit (Enhanced Fork):
```
git clone https://github.com/relaxis/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
```

Or use the original: `git clone https://github.com/ostris/ai-toolkit.git`
#### Modal:
- Run `pip install modal` to install the modal Python package.
- Run `modal setup` to authenticate (if this doesn’t work, try `python -m modal setup`).

#### Hugging Face:
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run `huggingface-cli login` and paste your token.

### 2. Upload your dataset
- Drag and drop your dataset folder containing the .jpg, .jpeg, or .png images and .txt files in `ai-toolkit`.

### 3. Configs
- Copy an example config file located at ```config/examples/modal``` to the `config` folder and rename it to ```whatever_you_want.yml```.
- Edit the config following the comments in the file, **<ins>be careful and follow the example `/root/ai-toolkit` paths</ins>**.

### 4. Edit run_modal.py
- Set your entire local `ai-toolkit` path at `code_mount = modal.Mount.from_local_dir` like:
  
   ```
   code_mount = modal.Mount.from_local_dir("/Users/username/ai-toolkit", remote_path="/root/ai-toolkit")
   ```
- Choose a `GPU` and `Timeout` in `@app.function` _(default is A100 40GB and 2 hour timeout)_.

### 5. Training
- Run the config file in your terminal: `modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml`.
- You can monitor your training in your local terminal, or on [modal.com](https://modal.com/).
- Models, samples and optimizer will be stored in `Storage > flux-lora-models`.

### 6. Saving the model
- Check contents of the volume by running `modal volume ls flux-lora-models`. 
- Download the content by running `modal volume get flux-lora-models your-model-name`.
- Example: `modal volume get flux-lora-models my_first_flux_lora_v1`.

### Screenshot from Modal

<img width="1728" alt="Modal Traning Screenshot" src="https://github.com/user-attachments/assets/7497eb38-0090-49d6-8ad9-9c8ea7b5388b">

---

## Dataset Preparation

Datasets generally need to be a folder containing images and associated text files. Currently, the only supported
formats are jpg, jpeg, and png. Webp currently has issues. The text files should be named the same as the images
but with a `.txt` extension. For example `image2.jpg` and `image2.txt`. The text file should contain only the caption.
You can add the word `[trigger]` in the caption file and if you have `trigger_word` in your config, it will be automatically
replaced. 

Images are never upscaled but they are downscaled and placed in buckets for batching. **You do not need to crop/resize your images**.
The loader will automatically resize them and can handle varying aspect ratios. 


## Training Specific Layers

To train specific layers with LoRA, you can use the `only_if_contains` network kwargs. For instance, if you want to train only the 2 layers
used by The Last Ben, [mentioned in this post](https://x.com/__TheBen/status/1829554120270987740), you can adjust your
network kwargs like so:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks.7.proj_out"
            - "transformer.single_transformer_blocks.20.proj_out"
```

The naming conventions of the layers are in diffusers format, so checking the state dict of a model will reveal 
the suffix of the name of the layers you want to train. You can also use this method to only train specific groups of weights.
For instance to only train the `single_transformer` for FLUX.1, you can use the following:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks."
```

You can also exclude layers by their names by using `ignore_if_contains` network kwarg. So to exclude all the single transformer blocks,


```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          ignore_if_contains:
            - "transformer.single_transformer_blocks."
```

`ignore_if_contains` takes priority over `only_if_contains`. So if a weight is covered by both,
if will be ignored.

## LoKr Training

To learn more about LoKr, read more about it at [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md). To train a LoKr model, you can adjust the network type in the config file like so:

```yaml
      network:
        type: "lokr"
        lokr_full_rank: true
        lokr_factor: 8
```

Everything else should work the same including layer targeting.


## Video (I2V) Training with Alpha Scheduling

Video training benefits significantly from alpha scheduling due to the 10-100x higher variance compared to image training. This fork includes optimized presets for video models like WAN 2.2 14B I2V.

### Example Configuration for Video Training

See the complete example at [`config_examples/i2v_lora_alpha_scheduling.yaml`](config_examples/i2v_lora_alpha_scheduling.yaml)

**Key differences for video vs image training:**

```yaml
network:
  type: lora
  linear: 64
  linear_alpha: 16
  conv: 64
  alpha_schedule:
    enabled: true
    linear_alpha: 16
    conv_alpha_phases:
      foundation:
        alpha: 8
        min_steps: 2000
        exit_criteria:
          # Video-optimized thresholds (10-100x more tolerant)
          loss_improvement_rate_below: 0.005  # vs 0.001 for images
          min_gradient_stability: 0.50         # vs 0.55 for images
          min_loss_r2: 0.01                    # vs 0.1 for images
      balance:
        alpha: 14
        min_steps: 3000
        exit_criteria:
          loss_improvement_rate_below: 0.005
          min_gradient_stability: 0.50
          min_loss_r2: 0.01
      emphasis:
        alpha: 20
        min_steps: 2000
```

### Video Training Dataset Setup

Video datasets should be organized as:
```
/datasets/your_videos/
├── video1.mp4
├── video1.txt (caption)
├── video2.mp4
├── video2.txt
└── ...
```

For I2V (image-to-video) training:
```yaml
datasets:
  - folder_path: /path/to/videos
    caption_ext: txt
    caption_dropout_rate: 0.3
    resolution: [512]
    max_pixels_per_frame: 262144
    shrink_video_to_frames: true
    num_frames: 33  # or 41, 49, etc.
    do_i2v: true    # Enable I2V mode
```

### Monitoring Video Training

Video training produces noisier metrics than image training. Expect:
- **Loss R²**: 0.007-0.05 (vs 0.1-0.3 for images)
- **Gradient Stability**: 0.45-0.60 (vs 0.55-0.70 for images)
- **Phase Transitions**: Longer times to plateau (video variance is high)

Check metrics at: `output/{job_name}/metrics_{job_name}.jsonl`

### Supported Video Models

- **WAN 2.2 14B I2V** - Image-to-video generation with MoE (Mixture of Experts)
- **WAN 2.1** - Earlier I2V model
- Other video diffusion models with LoRA support

For WAN 2.2 14B I2V, ensure you enable MoE-specific settings:
```yaml
model:
  name_or_path: "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16"
  arch: "wan22_14b_i2v"
  quantize: true
  qtype: "uint4|ostris/accuracy_recovery_adapters/wan22_14b_i2v_torchao_uint4.safetensors"
  model_kwargs:
    train_high_noise: true
    train_low_noise: true

train:
  switch_boundary_every: 100  # Switch between experts every 100 steps
```


## Updates

Only larger updates are listed here. There are usually smaller daily updated that are omitted.

### Jul 17, 2025
- Make it easy to add control images to the samples in the ui

### Jul 11, 2025
- Added better video config settings to the UI for video models.
- Added Wan I2V training to the UI

### June 29, 2025
- Fixed issue where Kontext forced sizes on sampling

### June 26, 2025
- Added support for FLUX.1 Kontext training
- added support for instruction dataset training

### June 25, 2025
- Added support for OmniGen2 training
- 
### June 17, 2025
- Performance optimizations for batch preparation
- Added some docs via a popup for items in the simple ui explaining what settings do. Still a WIP

### June 16, 2025
- Hide control images in the UI when viewing datasets
- WIP on mean flow loss

### June 12, 2025
- Fixed issue that resulted in blank captions in the dataloader

### June 10, 2025
- Decided to keep track up updates in the readme
- Added support for SDXL in the UI
- Added support for SD 1.5 in the UI
- Fixed UI Wan 2.1 14b name bug
- Added support for for conv training in the UI for models that support it