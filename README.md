![Logo](/icon.png)

# Emosic
Code for the paper: [CNN-based Facial Affect Analysis on Mobile Devices](TODO).

Music recommendation app based on user affect using CNNs for emotion classification and valence/arousal regression. Song recommendations made using the [Spotify Web API](https://developer.spotify.com/web-api/). Intended as a proof-of-concept that emotionally intelligent user interfaces are now feasbile on todays high-spec mobile phones.

CNNs are trained on [AffectNet](http://mohammadmahoor.com/affectnet/) using [Keras](https://keras.io) and deployed to iOS by conversion to coreML using [coremltools](https://github.com/apple/coremltools). Networks are designed to minimise space consumption, but retain near state-of-the-art performance on AffectNet.
