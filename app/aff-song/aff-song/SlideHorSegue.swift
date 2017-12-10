//
//  SlideHorSegue.swift
//  aff-song
//
//  Created by Charlie Hewitt on 10/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import UIKit

class SlideHorSegue: UIStoryboardSegue {
    override func perform() {
        //set the ViewControllers for the animation
        let sourceView = self.source.view as UIView!
        let destinationView = self.destination.view as UIView!
        let window = UIApplication.shared.delegate?.window!
        window?.insertSubview(destinationView!, belowSubview: sourceView!)
        destinationView?.center = CGPoint(x: (sourceView?.center.x)! + (destinationView?.frame.width)!, y: (sourceView?.center.y)!)
        UIView.animate(withDuration: 0.4,
                       animations: {
                        destinationView?.center = CGPoint(x: (sourceView?.center.x)!, y: (sourceView?.center.y)!)
        }, completion: {
            (value: Bool) in
            destinationView?.removeFromSuperview()
            if let navController = self.destination.navigationController {
                navController.popToViewController(self.destination, animated: false)
            }
        })
    }
}
