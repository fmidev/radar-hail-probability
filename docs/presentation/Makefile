TEX = agenttiohjelmointi_esitys.tex
PDF = $(TEX:.tex=.pdf)

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX)
	pdflatex -shell-escape $<
	pdflatex -shell-escape $<

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc
	rm -rf _minted-*
