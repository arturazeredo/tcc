#lang racket


(require plot)
(require json)
(require "dados.rkt")

(define (read-json-file s)
  (call-with-input-file s
    (lambda (in) (read-json in))))

(define (distance-scale p1 p2)
  ;; Calucule scale distance primitive version
  (sqrt (+ (expt (- (cadr p2) (cadr p1)) 2)
           (expt (- (car p2) (car p1)) 2) )))

(define (distance-point points start-point end-point)
  ;; Distance points
  (distance-scale (vector-ref points start-point) (vector-ref points end-point)) )

(define (get-x points index)
  ;; Get x Point with index
  (car (vector-ref points index)))

(define (get-y points index)
  ;; Get y Point with index  
  (cadr (vector-ref points index)))


(define (distance-x points start-point end-point)
  ;; Calcule distance X points
  (- (get-x points end-point) (get-x points start-point)))


(define (distance-y points start-point end-point)
  ;; Calcule distance Y points
  (- (get-y points end-point) (get-y points end-point)))


(define (prop-face-x points width [px-i 0] [px-f 16])
  (/ (distance-x points px-i px-f) width))


(define (prop-face-y points height [py-i 19] [py-f 24])
  (/ (distance-y points py-i py-f) height))


(define (calcule-face-distance points [direct 'left])
  (cond ((eq? direct 'left)
         (values (distance-point points 17 29) #t))
        ((eq? direct 'right)
         (values (distance-point points 1 29) #t))
        (#t (values 0 #f) )))


(define (getall-points-x points)
 (for/vector ([i points]) (car i)))

(define (getall-points-y points)
  (for/vector ([i points]) (cadr i)))


(define (centroid-region points)
  (let ((min-x (vector-argmin car points))
        (max-x (vector-argmax car points))
        (min-y (vector-argmin cadr points))
        (max-y (vector-argmax cadr points)))
    (list (+ (car min-x) (/ (distance-scale min-x max-x) 2))
          (+ (cadr min-y)(/ (distance-scale min-y max-y) 2)))))


(define (diff-contour old-cnt new-cnt)
  (let ((old-central (centroid-region old-cnt))
        (new-central (centroid-region new-cnt)))
    (list (- (car new-central) (car old-central))
          (- (cadr new-central) (cadr old-central))) ))


(define (calcule-diff-points old-points new-points)
  (for/list ([pi old-points]
             [pf new-points])
    (list (- (car pf) (car pi))
          (- (cadr pf) (cadr pi))) ))


(define (diff-points old-points new-points [points-filter '(1 38 367)])
  (if (empty? points-filter)
      (calcule-diff-points old-points new-points)
      (calcule-diff-points (map (lambda (x) (vector-ref old-points x)) points-filter)
                           (map (lambda (x) (vector-ref new-points x)) points-filter)) ))
                                          

(define (rotate-180 pts)
  (list->vector (map (lambda (x) (list (car x) (* (cadr x) -1))) (vector->list pts))))
  

(define (visualize-points pts)
  (plot (points (map (lambda (x) (list (car x) (* (cadr x) -1))) (vector->list pts)))));; #:width 1024 #:height 1024))

(define (visualize-centroid pts)
  (plot (list (points (vector (centroid-region pts)) #:color 'red #:sym 'fullcircle2) (points pts))))

(define (visualize-in-mask pts point)
  (plot (list (points (vector point) #:color 'red #:sym 'fullcircle2) (points points-mask))))


#|
> (hash-iterate-first *x*)
0
> (hash-iterate-key *x* 0)
'landmark
> (hash-iterate-key *x* 1)
'initialTimeAnimation
> (hash-iterate-key *x* 2)
'token
> (hash-ref *x* 'token)
"4410d99cefe57ec2c2cdbd3f1d5cf862bb4fb6f8"
> 
|#

