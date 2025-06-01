
(ql:quickload :cl-json)

(defun read-logs (path-logs)
  (let ((in (open path-logs :if-does-not-exist nil)))
    (when in (loop for line = (read-line in nil)
		   while line do (format t "~a~%" line))
	  (close in))))



(defun decode-string-json (path-logs)
  (with-input-from-string
   (json:decode-json s)))

  
