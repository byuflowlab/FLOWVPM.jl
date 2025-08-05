using Documenter, FLOWVPM

DocMeta.setdocmeta!(FLOWVPM, :DocTestSetup, :(using FLOWVPM); recursive=true)

makedocs(sitename="FLOWVPM.jl",
        modules=[FLOWVPM],
        format=Documenter.HTML(;
        canonical="https://flow.byu.edu/FLOWVPM",
        edit_link="main",
        assets=String[]),
        pages=[
        "Introduction" => "index.md",
        "Quick Start" => "quickstart.md",
        "Guided Examples" => "guided.md",
        "Advanced Usage" => "advanced.md",
        "Theory" => "rVPM.md",
        "Reference" => "reference.md",
        ],
        checkdocs=:none
    )

deploydocs(;
    repo="github.com/byuflowlab/FLOWVPM",
    devbranch="main",
)
