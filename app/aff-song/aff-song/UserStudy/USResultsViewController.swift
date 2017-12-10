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
    var data : [String]?
    @IBOutlet weak var StarRatingView: HCSStarRatingView!
    
    override func viewDidLoad() {
        var runId = UserDefaults.standard.integer(forKey: "RunNumber")
        data!.append("\(runId)")
        data!.append("\(emotion!)")
        super.viewDidLoad()
    }
    
    @IBAction func ContinueButtonClicked(_ sender: Any) {
        data!.append("\(StarRatingView.value)")
    }
    
    override func getAffect(image: UIImage?) -> (Double, Double) {
        let (valence, arousal) : (Double, Double) = super.getAffect(image: image)
        data!.append("\(valence)")
        data!.append("\(arousal)")
        return (valence, arousal)
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

}
