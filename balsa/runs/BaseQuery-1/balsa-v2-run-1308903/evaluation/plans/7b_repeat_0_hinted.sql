/*+ HashJoin(it pi ci t lt ml n an)
 HashJoin(it pi ci t lt ml n)
 HashJoin(pi ci t lt ml n)
 HashJoin(ci t lt ml n)
 HashJoin(ci t lt ml)
 HashJoin(t lt ml)
 HashJoin(lt ml)
 SeqScan(it)
 SeqScan(pi)
 SeqScan(ci)
 SeqScan(t)
 SeqScan(lt)
 IndexScan(ml)
 IndexScan(n)
 IndexScan(an)
 Leading(((it (pi ((ci (t (lt ml))) n))) an)) */
SELECT MIN(n.name) AS of_person,
       MIN(t.title) AS biography_movie
FROM aka_name AS an,
     cast_info AS ci,
     info_type AS it,
     link_type AS lt,
     movie_link AS ml,
     name AS n,
     person_info AS pi,
     title AS t
WHERE an.name LIKE '%a%'
  AND it.info ='mini biography'
  AND lt.link ='features'
  AND n.name_pcode_cf LIKE 'D%'
  AND n.gender='m'
  AND pi.note ='Volker Boehm'
  AND t.production_year BETWEEN 1980 AND 1984
  AND n.id = an.person_id
  AND n.id = pi.person_id
  AND ci.person_id = n.id
  AND t.id = ci.movie_id
  AND ml.linked_movie_id = t.id
  AND lt.id = ml.link_type_id
  AND it.id = pi.info_type_id
  AND pi.person_id = an.person_id
  AND pi.person_id = ci.person_id
  AND an.person_id = ci.person_id
  AND ci.movie_id = ml.linked_movie_id;

