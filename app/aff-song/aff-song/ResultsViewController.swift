//
//  ResultsViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 08/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ResultsViewController: AffUIViewController, UITableViewDataSource, UITableViewDelegate {
    
    var faceImage : UIImage?
    var tracks : [[String]]?
    @IBOutlet var ImageView: UIImageView!
    @IBOutlet weak var TrackTable: UITableView!
    @IBOutlet weak var EmotionLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        ImageView.image = faceImage
        let (valence, arousal, emotion, _) = getAffect(image: faceImage)
        getSongs(valence: valence, arousal: arousal, emotion: emotion, callback: {tracks in
            self.tracks = tracks
            DispatchQueue.main.async {
                self.TrackTable.reloadData()
            }
        })
    }
    
    @IBAction func backButtonClick(_ sender: Any) {
        dismiss(animated: true, completion: nil)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func getSongs(valence : Double, arousal: Double, emotion: Int, callback: @escaping ([[String]])->Void){
        guard let authKey = UserDefaults.standard.string(forKey: "SpotifyAuthToken") else {
            fatalError("Spotify auth failed")
        }
        var genres : String?
        // "acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime", "black-metal", "bluegrass", "blues", "bossanova", "brazil", "breakbeat", "british", "cantopop", "chicago-house", "children", "chill", "classical", "club", "comedy", "country", "dance", "dancehall", "death-metal", "deep-house", "detroit-techno", "disco", "disney", "drum-and-bass", "dub", "dubstep", "edm", "electro", "electronic", "emo", "folk", "forro", "french", "funk", "garage", "german", "gospel", "goth", "grindcore", "groove", "grunge", "guitar", "happy", "hard-rock", "hardcore", "hardstyle", "heavy-metal", "hip-hop", "holidays", "honky-tonk", "house", "idm", "indian", "indie", "indie-pop", "industrial", "iranian", "j-dance", "j-idol", "j-pop", "j-rock", "jazz", "k-pop", "kids", "latin", "latino", "malay", "mandopop", "metal", "metal-misc", "metalcore", "minimal-techno", "movies", "mpb", "new-age", "new-release", "opera", "pagode", "party", "philippines-opm", "piano", "pop", "pop-film", "post-dubstep", "power-pop", "progressive-house", "psych-rock", "punk", "punk-rock", "r-n-b", "rainy-day", "reggae", "reggaeton", "road-trip", "rock", "rock-n-roll", "rockabilly", "romance", "sad", "salsa", "samba", "sertanejo", "show-tunes", "singer-songwriter", "ska", "sleep", "songwriter", "soul", "soundtracks", "spanish", "study", "summer", "swedish", "synth-pop", "tango", "techno", "trance", "trip-hop", "turkish", "work-out", "world-music"
        switch emotion {
        case 0:
            // "Neutral"
            genres = "jazz,ambient,reggae,indie,alt-rock"
            break
        case 1:
            // "happy"
            genres = "dance,pop,rock,show-tunes,party"
            break
        case 2:
            // "sad"
            genres = "sad,classical,folk,songwriter,blues"
            break
        case 3:
            // "surised"
            genres = "funk,electronic,techno,disco,ska"
            break
        case 4:
            // "afraid"
            genres = "classical,soul,piano,"
            break
        case 5:
            // "disgusted"
            genres = "drum-and-bass,dub,edm,house,club"
            break
        case 6:
            // "Angry"
            genres = "metal,grunge,punk,industrial,garage"
            break
        case 7:
            // "Contemptful"
            genres = "blues,soul,country"
            break
        default:
            genres = "pop,rock"
            // "Error"
        }
        var mode = 0
        if valence >= 0 {
            mode = 1
        }
        // map v/a to [0,1]
        let urlPath: String = "https://api.spotify.com/v1/recommendations?seed_genres=\(genres!)&target_valence=\((valence+1)/2)&target_energy=\((arousal+1)/2)&mode=\(mode)&limit=5&market=GB"
        print(urlPath)
        var request: URLRequest = URLRequest(url: URL(string: urlPath)!)
        request.httpMethod = "GET"
        request.addValue("Bearer " + authKey, forHTTPHeaderField: "Authorization")
        let session = URLSession.shared
        session.dataTask(with: request) {data, response, err in
            do {
                let jsonResult = try JSONSerialization.jsonObject(with: data!, options: []) as! [String:[Any]]
                var tracks : [[String]] = []
                for track in jsonResult["tracks"]! {
                    let track_d = track as! [String:Any]
                    let artists = track_d["artists"] as! [[String:Any]]
                    tracks.append([track_d["name"] as! String, artists[0]["name"] as! String, track_d["uri"] as! String])
                }
                callback(tracks)
            } catch let error {
                print(error.localizedDescription)
            }
            }.resume()
    }
    
    
    func getAffect(image: UIImage?) -> (Double, Double, Int, [String:Double]) {
        // 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face
        let pixelBuffer = image?.pixelBuffer(width: 128, height: 128)
        let classifier = MobAffNetC()
        let regressor = MobAffNetR()
        guard let predictedEmotion = try? classifier.prediction(image: pixelBuffer!),
              let predictedValArr = try? regressor.prediction(image: pixelBuffer!) else {
            fatalError("Unexpected runtime error.")
        }
        var valence = predictedValArr.valence_arousal[0].doubleValue
        var arousal = predictedValArr.valence_arousal[1].doubleValue
        let emotion = Int(predictedEmotion.emotion)!
        let emotion_p = predictedEmotion.emotion_p
        var emotionText : String?
        switch emotion {
        case 0:
            emotionText = "Neutral"
            break
        case 1:
            emotionText = "Happy"
            break
        case 2:
            emotionText = "Sad"
            break
        case 3:
            emotionText = "Surprised"
            break
        case 4:
            emotionText = "Afraid"
            break
        case 5:
            emotionText = "Disgusted"
            break
        case 6:
            emotionText = "Angry"
            break
        case 7:
            emotionText = "Contemptuous"
            break
        default:
            emotionText = "Unknown"
        }
        if EmotionLabel != nil {
            EmotionLabel.text =  "Predicted Emotion: \(emotionText!)\n" + String(format: "Valence: %.2f    Arousal: %.2f", valence, arousal)
        }
        valence = max(-1, min(1, valence))
        arousal = max(-1, min(1, arousal))
        return (valence, arousal, emotion, emotion_p)
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return self.tracks?.count ?? 0
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell:UITableViewCell = UITableViewCell(style:.subtitle, reuseIdentifier:"cell")
        cell.textLabel?.text = self.tracks![indexPath.row][0]
        cell.textLabel?.textColor = UIColor.white
        cell.detailTextLabel?.text = self.tracks![indexPath.row][1]
        cell.detailTextLabel?.textColor = UIColor.white
        cell.backgroundColor = .clear
        cell.selectionStyle = .none
        cell.backgroundColor = .clear
        return cell
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        UIApplication.shared.open(URL(string: self.tracks![indexPath.row][2])!, options: [:], completionHandler: nil)
    }
}

extension UIImage {
    public func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var maybePixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue]
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         width,
                                         height,
                                         kCVPixelFormatType_32ARGB,
                                         attrs as CFDictionary,
                                         &maybePixelBuffer)
        
        guard status == kCVReturnSuccess, let pixelBuffer = maybePixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer)
        
        guard let context = CGContext(data: pixelData,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
            else {
                return nil
        }
        
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer
    }
}
