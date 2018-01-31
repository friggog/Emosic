    //
//  USResultsViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 10/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import Foundation

class USResultsViewController: ResultsViewController, UINavigationControllerDelegate {
    
    var emotion: Int?
    var emotionLabel : Int?
    var data : [String]?
    var ratingMade = false
    @IBOutlet weak var StarRatingView: HCSStarRatingView!
    
    override func viewDidLoad() {
        let runId = UserDefaults.standard.integer(forKey: "RunNumber")
        data!.append("\(runId)")
        data!.append("\(emotion!)")
        data!.append("\(emotionLabel!)")
        super.viewDidLoad()
    }
    
    @IBAction func RatingMade(_ sender: Any) {
        ratingMade = true
    }
    
    @IBAction func ContinueButtonClicked(_ sender: Any) {
        if ratingMade {
            data!.append("\(StarRatingView.value)")
            if emotion == 0 {
                data!.append("0.0")
                data!.append("0.0")
                let vc = USAnnotateViewController()
                vc.data = data
                vc.saveData()
                performSegue(withIdentifier: "usNeutralSegue", sender: self)
            }
            else {
                performSegue(withIdentifier: "usAnnotateSegue", sender: self)
            }
        }
        else {
            let alert = UIAlertController(title: "Rating Required", message: "Please rate the recommended songs.", preferredStyle: UIAlertControllerStyle.alert)
            alert.addAction(UIAlertAction(title: "OK", style: UIAlertActionStyle.cancel, handler: nil))
            self.present(alert, animated: true, completion: nil)
        }
    }
    
    override func getAffect(image: UIImage?) -> (Double, Double, Int, [String:Double]) {
        let (valence, arousal, emotion, emotion_p) = super.getAffect(image: image)
        data!.append("\(valence)")
        data!.append("\(arousal)")
        data!.append("\(emotion)")
        for i in 0 ... 7 {
            data!.append("\(emotion_p["\(i)"] ?? 0)")
        }
        return (valence, arousal, emotion, emotion_p)
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "usAnnotateSegue" {
            if let nextViewController = segue.destination as? USAnnotateViewController {
                nextViewController.emotion = self.emotion
                nextViewController.faceImage = self.faceImage
                nextViewController.data = self.data
            }
        }
    }
    
    @IBAction func cancelButtonClicked(_ sender: Any) {
        let alert = UIAlertController(title: "Exit User Study", message: "Are you sure you would like to exit the study?", preferredStyle: UIAlertControllerStyle.alert)
        alert.addAction(UIAlertAction(title: "Exit", style: UIAlertActionStyle.destructive, handler: { action in
            self.navigationController?.dismiss(animated: true, completion: nil)
        }))
        alert.addAction(UIAlertAction(title: "Cancel", style: UIAlertActionStyle.cancel, handler: nil))
        self.present(alert, animated: true, completion: nil)
    }
}
